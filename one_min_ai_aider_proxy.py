
#!/usr/bin/env python3
"""
one_min_ai_aider_proxy.py

Proxy local OpenAI-compatible pour utiliser Aider avec l'API 1min.ai CODE_GENERATOR.

Endpoints exposés :
  GET  /v1/models
  POST /v1/chat/completions
  POST /v1/responses

Pourquoi /v1/responses ?
  Les versions récentes d'Aider/LiteLLM utilisent souvent l'API OpenAI Responses
  pour les modèles GPT-5/Codex. Sans cet endpoint, Aider tente POST /v1/responses
  et reçoit une 404.

Variables d'environnement :
  ONE_MIN_API_KEY              Clé API 1min.ai obligatoire
  ONE_MIN_MODEL                Modèle 1min.ai par défaut, ex: gpt-4o, gpt-5.1-codex-mini
  ONE_MIN_CONVERSATION_ID      Conversation ID, défaut: CODE_GENERATOR
  ONE_MIN_WEB_SEARCH           true/false, défaut: false
  ONE_MIN_NUM_OF_SITE          1-10, optionnel
  ONE_MIN_MAX_WORD             100-10000, optionnel
  ONE_MIN_PROXY_PREAMBLE       Préambule ajouté au prompt envoyé au modèle, optionnel
  ONE_MIN_TIMEOUT              Timeout HTTP vers 1min.ai, défaut: 180
  ONE_MIN_LOG_PROMPTS          true/false, défaut: false

Exemple Windows Git Bash :
  cd ~
  source aider_env/Scripts/activate
  export ONE_MIN_API_KEY="..."
  export ONE_MIN_MODEL="gpt-5.1-codex-mini"
  python ./one_min_ai_aider_proxy.py --port 8787

Dans un autre terminal, depuis ton projet :
  source ~/aider_env/Scripts/activate
  winpty aider \\
    --model openai/gpt-5.1-codex-mini \\
    --openai-api-base http://127.0.0.1:8787/v1 \\
    --openai-api-key dummy \\
    --no-auto-commits
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple


ONE_MIN_ENDPOINT = "https://api.1min.ai/api/features"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def normalize_model(model: Optional[str]) -> str:
    """
    Aider/LiteLLM peut envoyer :
      - gpt-4o
      - openai/gpt-4o
      - openai/gpt-5.1-codex-mini

    L'API 1min.ai attend normalement le nom brut du modèle.
    """
    default_model = os.getenv("ONE_MIN_MODEL", "gpt-4o")

    if not model:
        return default_model

    model = str(model).strip()

    prefixes = [
        "openai/",
        "openai_chat/",
        "openai_responses/",
        "1min/",
        "one_min/",
        "one-min/",
    ]

    for prefix in prefixes:
        if model.startswith(prefix):
            return model[len(prefix):]

    return model


def content_part_to_text(part: Any) -> str:
    """
    Convertit différents formats de contenu OpenAI en texte simple.
    """
    if part is None:
        return ""

    if isinstance(part, str):
        return part

    if isinstance(part, dict):
        part_type = part.get("type")

        if part_type in {"text", "input_text", "output_text"}:
            return str(part.get("text", ""))

        if part_type == "image_url":
            return "[image_url non supportée par ce proxy texte]"

        if "text" in part:
            return str(part.get("text", ""))

        return json.dumps(part, ensure_ascii=False)

    return str(part)


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            part_text
            for part_text in (content_part_to_text(part) for part in content)
            if part_text
        )

    return json.dumps(content, ensure_ascii=False)


def flatten_chat_messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Transforme les messages Chat Completions en prompt texte.
    """
    chunks: List[str] = []
    add_common_preamble(chunks)

    for index, message in enumerate(messages):
        role = str(message.get("role", "user")).upper()
        name = message.get("name")
        content = message_content_to_text(message.get("content"))

        title = f"### MESSAGE {index + 1} - ROLE: {role}"
        if name:
            title += f" - NAME: {name}"

        chunks.append(title)
        chunks.append(content)

    return "\n\n".join(chunks).strip()


def flatten_responses_input_to_prompt(body: Dict[str, Any]) -> str:
    """
    Transforme le body OpenAI Responses API en prompt texte.

    Formats fréquents :
      {
        "model": "...",
        "instructions": "...",
        "input": "..."
      }

      {
        "model": "...",
        "input": [
          {"role": "user", "content": [{"type": "input_text", "text": "..."}]}
        ]
      }
    """
    chunks: List[str] = []
    add_common_preamble(chunks)

    instructions = body.get("instructions")
    if instructions:
        chunks.append("### INSTRUCTIONS")
        chunks.append(message_content_to_text(instructions))

    previous_response_id = body.get("previous_response_id")
    if previous_response_id:
        chunks.append("### PREVIOUS_RESPONSE_ID")
        chunks.append(str(previous_response_id))

    input_value = body.get("input")

    if isinstance(input_value, str):
        chunks.append("### INPUT")
        chunks.append(input_value)

    elif isinstance(input_value, list):
        for index, item in enumerate(input_value):
            if isinstance(item, dict):
                role = str(item.get("role", item.get("type", "user"))).upper()
                content = item.get("content", "")

                chunks.append(f"### INPUT ITEM {index + 1} - ROLE: {role}")
                chunks.append(message_content_to_text(content))
            else:
                chunks.append(f"### INPUT ITEM {index + 1}")
                chunks.append(message_content_to_text(item))

    elif input_value is not None:
        chunks.append("### INPUT")
        chunks.append(json.dumps(input_value, ensure_ascii=False))

    return "\n\n".join(chunks).strip()


def add_common_preamble(chunks: List[str]) -> None:
    """
    Ajoute un préambule minimal. Aider fournit déjà ses règles détaillées.
    """
    custom_preamble = os.getenv("ONE_MIN_PROXY_PREAMBLE", "").strip()

    if custom_preamble:
        chunks.append("### ADAPTER PREAMBLE")
        chunks.append(custom_preamble)

    chunks.append(
        "### IMPORTANT POUR LE MODELE\n"
        "Tu réponds à un agent de développement de code en terminal, généralement Aider.\n"
        "Respecte strictement les consignes de format présentes dans la conversation.\n"
        "Si Aider demande un format de diff, de patch ou d'édition précis, utilise exactement ce format.\n"
        "Ne produis pas de Markdown décoratif autour des patchs si les instructions demandent un format brut.\n"
        "Ne prétends pas avoir lu des fichiers qui ne sont pas fournis dans le contexte.\n"
        "Ne fais pas de modifications non demandées."
    )


def extract_1min_text(response_json: Dict[str, Any]) -> str:
    """
    Extrait le texte utile depuis le format de réponse 1min.ai.

    Format documenté typique :
      {
        "aiRecord": {
          "aiRecordDetail": {
            "resultObject": ["..."]
          }
        }
      }
    """
    try:
        detail = response_json.get("aiRecord", {}).get("aiRecordDetail", {})
        result = detail.get("resultObject")

        if isinstance(result, list):
            parts = []
            for item in result:
                if isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            return "\n".join(parts).strip()

        if isinstance(result, str):
            return result.strip()

        if result is not None:
            return json.dumps(result, ensure_ascii=False, indent=2)

        # Fallbacks possibles si 1min.ai change légèrement le format.
        for key in ("result", "text", "content", "message"):
            if key in detail:
                return message_content_to_text(detail[key]).strip()

        return json.dumps(response_json, ensure_ascii=False, indent=2)

    except Exception:
        return json.dumps(response_json, ensure_ascii=False, indent=2)


def call_1min_api(prompt: str, model: str, timeout_seconds: int) -> str:
    api_key = os.getenv("ONE_MIN_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Variable d'environnement ONE_MIN_API_KEY manquante. "
            "Exporte-la AVANT de lancer le proxy."
        )

    prompt_object: Dict[str, Any] = {
        "prompt": prompt,
        "webSearch": env_bool("ONE_MIN_WEB_SEARCH", False),
    }

    num_of_site = env_int("ONE_MIN_NUM_OF_SITE")
    max_word = env_int("ONE_MIN_MAX_WORD")

    if num_of_site is not None:
        prompt_object["numOfSite"] = num_of_site

    if max_word is not None:
        prompt_object["maxWord"] = max_word

    payload = {
        "type": "CODE_GENERATOR",
        "model": model,
        "conversationId": os.getenv("ONE_MIN_CONVERSATION_ID", "CODE_GENERATOR"),
        "promptObject": prompt_object,
    }

    if env_bool("ONE_MIN_LOG_PROMPTS", False):
        print("----- PROMPT ENVOYE A 1MIN.AI -----", file=sys.stderr)
        print(prompt, file=sys.stderr)
        print("----- FIN PROMPT -----", file=sys.stderr)

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        ONE_MIN_ENDPOINT,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "API-KEY": api_key,
            "User-Agent": os.getenv(
                "ONE_MIN_USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            "Connection": "close",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            return extract_1min_text(parsed)

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Erreur HTTP 1min.ai {exc.code}: {error_body}") from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(f"Erreur réseau vers 1min.ai: {exc}") from exc


def openai_error_body(message: str, status: int = 500) -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "one_min_ai_proxy_error",
            "param": None,
            "code": status,
        }
    }


def chat_completion_response(model: str, content: str) -> Dict[str, Any]:
    now = int(time.time())
    approx_completion_tokens = max(1, len(content.split()))

    return {
        "id": f"chatcmpl-1min-{now}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": approx_completion_tokens,
            "total_tokens": approx_completion_tokens,
        },
    }


def responses_response(model: str, content: str) -> Dict[str, Any]:
    """
    Réponse compatible OpenAI Responses API.

    Structure importante pour les clients :
      - output[] contient un message assistant
      - content[] contient un item type output_text
      - output_text est aussi fourni comme raccourci pratique
    """
    now = int(time.time())
    response_id = f"resp_1min_{now}"
    approx_completion_tokens = max(1, len(content.split()))

    return {
        "id": response_id,
        "object": "response",
        "created_at": now,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": [
            {
                "id": f"msg_1min_{now}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ],
        "output_text": content,
        "parallel_tool_calls": False,
        "previous_response_id": None,
        "reasoning": None,
        "store": False,
        "temperature": None,
        "text": {
            "format": {
                "type": "text",
            }
        },
        "tool_choice": "none",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 0,
            "output_tokens": approx_completion_tokens,
            "total_tokens": approx_completion_tokens,
        },
        "user": None,
        "metadata": {},
    }


def sse_send(handler: BaseHTTPRequestHandler, event: Optional[str], data: Dict[str, Any]) -> None:
    if event:
        handler.wfile.write(f"event: {event}\n".encode("utf-8"))
    handler.wfile.write(("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode("utf-8"))
    handler.wfile.flush()


def stream_chat_completion(handler: BaseHTTPRequestHandler, model: str, content: str) -> None:
    now = int(time.time())
    completion_id = f"chatcmpl-1min-{now}"

    chunks = [
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        },
    ]

    for chunk in chunks:
        handler.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8"))
        handler.wfile.flush()

    handler.wfile.write(b"data: [DONE]\n\n")
    handler.wfile.flush()


def stream_responses(handler: BaseHTTPRequestHandler, model: str, content: str) -> None:
    """
    Streaming Responses API minimal.
    """
    now = int(time.time())
    response_id = f"resp_1min_{now}"
    item_id = f"msg_1min_{now}"

    created_response = responses_response(model, "")
    created_response["id"] = response_id
    created_response["status"] = "in_progress"
    created_response["output"] = []
    created_response["output_text"] = ""

    sse_send(
        handler,
        "response.created",
        {
            "type": "response.created",
            "response": created_response,
        },
    )

    sse_send(
        handler,
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
    )

    sse_send(
        handler,
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            },
        },
    )

    sse_send(
        handler,
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": content,
        },
    )

    sse_send(
        handler,
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": content,
        },
    )

    sse_send(
        handler,
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": content,
                "annotations": [],
                "logprobs": [],
            },
        },
    )

    sse_send(
        handler,
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            },
        },
    )

    final_response = responses_response(model, content)
    final_response["id"] = response_id

    sse_send(
        handler,
        "response.completed",
        {
            "type": "response.completed",
            "response": final_response,
        },
    )

    handler.wfile.write(b"data: [DONE]\n\n")
    handler.wfile.flush()


class OneMinAiderProxyHandler(BaseHTTPRequestHandler):
    server_version = "OneMinAiderProxy/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(
            "[%s] %s\n"
            % (
                self.log_date_time_string(),
                fmt % args,
            )
        )

    def send_json(self, status: int, body: Dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json(status, openai_error_body(message, status))

    def read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")

        if not raw:
            return {}

        return json.loads(raw)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]

        if path in {"/", "/health", "/v1/health"}:
            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "one_min_ai_aider_proxy",
                    "endpoints": [
                        "/v1/models",
                        "/v1/chat/completions",
                        "/v1/responses",
                    ],
                },
            )
            return

        if path in {"/v1/models", "/models"}:
            default_model = os.getenv("ONE_MIN_MODEL", "gpt-4o")
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": default_model,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "1min.ai",
                        }
                    ],
                },
            )
            return

        self.send_error_json(404, f"Endpoint inconnu: {self.path}")

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]

        if path in {"/v1/chat/completions", "/chat/completions"}:
            self.handle_chat_completions()
            return

        if path in {"/v1/responses", "/responses"}:
            self.handle_responses()
            return

        self.send_error_json(404, f"Endpoint inconnu: {self.path}")

    def handle_chat_completions(self) -> None:
        try:
            body = self.read_json_body()
            messages = body.get("messages", [])

            if not isinstance(messages, list):
                self.send_error_json(400, "Le champ 'messages' doit être une liste.")
                return

            model = normalize_model(body.get("model"))
            stream = bool(body.get("stream", False))
            timeout_seconds = int(os.getenv("ONE_MIN_TIMEOUT", "180"))

            prompt = flatten_chat_messages_to_prompt(messages)
            result_text = call_1min_api(
                prompt=prompt,
                model=model,
                timeout_seconds=timeout_seconds,
            )

            if stream:
                self.send_sse_headers()
                stream_chat_completion(self, model, result_text)
                return

            self.send_json(200, chat_completion_response(model=model, content=result_text))

        except json.JSONDecodeError as exc:
            self.send_error_json(400, f"JSON invalide: {exc}")

        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(502, str(exc))

    def handle_responses(self) -> None:
        try:
            body = self.read_json_body()

            model = normalize_model(body.get("model"))
            stream = bool(body.get("stream", False))
            timeout_seconds = int(os.getenv("ONE_MIN_TIMEOUT", "180"))

            prompt = flatten_responses_input_to_prompt(body)
            result_text = call_1min_api(
                prompt=prompt,
                model=model,
                timeout_seconds=timeout_seconds,
            )

            if stream:
                self.send_sse_headers()
                stream_responses(self, model, result_text)
                return

            self.send_json(200, responses_response(model=model, content=result_text))

        except json.JSONDecodeError as exc:
            self.send_error_json(400, f"JSON invalide: {exc}")

        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(502, str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proxy OpenAI/Responses-compatible pour utiliser Aider avec 1min.ai CODE_GENERATOR."
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Adresse d'écoute. Défaut: 127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Port d'écoute. Défaut: 8787",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    server = ThreadingHTTPServer(
        (args.host, args.port),
        OneMinAiderProxyHandler,
    )

    print(
        f"Proxy 1min.ai pour Aider démarré sur http://{args.host}:{args.port}/v1",
        flush=True,
    )
    print(
        "Endpoints: GET /v1/models, POST /v1/chat/completions, POST /v1/responses",
        flush=True,
    )

    if not os.getenv("ONE_MIN_API_KEY"):
        print(
            "Attention: ONE_MIN_API_KEY n'est pas défini. "
            "Exporte-la AVANT de lancer le proxy.",
            file=sys.stderr,
            flush=True,
        )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du proxy.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
