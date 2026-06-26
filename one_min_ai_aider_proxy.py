
#!/usr/bin/env python3
"""
one_min_ai_aider_proxy.py - v2

Proxy local OpenAI-compatible pour utiliser Aider avec l'API 1min.ai.

Endpoints exposes :
  GET  /v1/models
  POST /v1/chat/completions
  POST /v1/responses

Base sur les faits confirmes par tests :
  - Endpoint        : POST https://api.1min.ai/api/features
  - type accepte    : CODE_GENERATOR (configurable via ONE_MIN_FEATURE_TYPE)
  - Auth            : header API-KEY
  - Reponse         : aiRecord.aiRecordDetail.resultObject

Variables d'environnement principales :
  ONE_MIN_API_KEY        Cle API 1min.ai (obligatoire)
  ONE_MIN_MODEL          Modele par defaut (defaut: gpt-4o)
  ONE_MIN_FEATURE_TYPE   Type de feature (defaut: CODE_GENERATOR)
  ONE_MIN_TIMEOUT        Timeout HTTP (defaut: 180)
  ONE_MIN_WEB_SEARCH     true/false (defaut: false)
  ONE_MIN_NUM_OF_SITE    int optionnel
  ONE_MIN_MAX_WORD       int optionnel
  ONE_MIN_STREAM_CHUNK   Taille des chunks SSE en caracteres (defaut: 40)
  ONE_MIN_LOG_PROMPTS    true/false : log prompt envoye (defaut: false)
  ONE_MIN_LOG_RAW        true/false : log reponse brute 1min.ai (defaut: false)
  ONE_MIN_DEHTML_OUTPUT  true/false : decode entites HTML (defaut: true)

Lancement :
  export ONE_MIN_API_KEY="..."
  export ONE_MIN_MODEL="gpt-4o"
  python one_min_ai_aider_proxy.py --port 8787

Cote Aider :
  aider --model openai/gpt-4o \\
    --openai-api-base http://127.0.0.1:8787/v1 \\
    --openai-api-key dummy \\
    --no-auto-commits
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


ONE_MIN_ENDPOINT = "https://api.1min.ai/api/features"


# --------------------------------------------------------------------------- #
# Helpers env
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Normalisation du modele
# --------------------------------------------------------------------------- #

def normalize_model(model: Optional[str]) -> str:
    default_model = os.getenv("ONE_MIN_MODEL", "gpt-4o")
    if not model:
        return default_model

    model = str(model).strip()
    prefixes = (
        "openai/", "openai_chat/", "openai_responses/",
        "1min/", "one_min/", "one-min/",
    )
    for prefix in prefixes:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


# --------------------------------------------------------------------------- #
# Conversion contenu -> texte
# --------------------------------------------------------------------------- #

def part_to_text(part: Any) -> str:
    if part is None:
        return ""
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        ptype = part.get("type")
        if ptype in {"text", "input_text", "output_text"}:
            return str(part.get("text", ""))
        if ptype == "image_url":
            return "[image non supportee]"
        if "text" in part:
            return str(part.get("text", ""))
        return json.dumps(part, ensure_ascii=False)
    return str(part)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            t for t in (part_to_text(p) for p in content) if t
        )
    return json.dumps(content, ensure_ascii=False)


def chat_messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Conversion minimale : on respecte les roles, sans preambule injecte.
    Aider fournit deja toutes ses consignes de format.
    """
    chunks: List[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        text = content_to_text(msg.get("content"))
        if not text:
            continue
        chunks.append(f"[{role}]\n{text}")
    return "\n\n".join(chunks).strip()


def responses_input_to_prompt(body: Dict[str, Any]) -> str:
    chunks: List[str] = []

    instructions = body.get("instructions")
    if instructions:
        chunks.append(f"[SYSTEM]\n{content_to_text(instructions)}")

    input_value = body.get("input")
    if isinstance(input_value, str):
        chunks.append(f"[USER]\n{input_value}")
    elif isinstance(input_value, list):
        for item in input_value:
            if isinstance(item, dict):
                role = str(item.get("role", item.get("type", "user"))).upper()
                chunks.append(f"[{role}]\n{content_to_text(item.get('content', ''))}")
            else:
                chunks.append(content_to_text(item))
    elif input_value is not None:
        chunks.append(json.dumps(input_value, ensure_ascii=False))

    return "\n\n".join(chunks).strip()


# --------------------------------------------------------------------------- #
# Nettoyage sortie (HTML escaping uniquement, leger)
# --------------------------------------------------------------------------- #

def html_unescape_repeated(text: str, max_rounds: int = 5) -> str:
    previous = text
    for _ in range(max_rounds):
        current = html.unescape(previous)
        if current == previous:
            return current
        previous = current
    return previous


def sanitize_output(text: str) -> str:
    if not isinstance(text, str):
        return text
    if env_bool("ONE_MIN_DEHTML_OUTPUT", True):
        text = html_unescape_repeated(text)
    if env_bool("ONE_MIN_UNESCAPE_MARKDOWN_OUTPUT", True):
        # Dict propre : plus de cles dupliquees.
        replacements = {
            "\\_": "_",
            "\\#": "#",
            "\\&": "&",
            "\\`": "`",
            "\\/": "/",
            "\\*": "*",
            "\\[": "[",
            "\\]": "]",
            "\\(": "(",
            "\\)": ")",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
    return text


# --------------------------------------------------------------------------- #
# Extraction de la reponse 1min.ai
# --------------------------------------------------------------------------- #

def extract_1min_text(response_json: Dict[str, Any]) -> str:
    detail = (
        response_json.get("aiRecord", {})
        .get("aiRecordDetail", {})
    )
    result = detail.get("resultObject")

    if isinstance(result, list):
        parts = [
            item if isinstance(item, str)
            else json.dumps(item, ensure_ascii=False)
            for item in result
        ]
        return "\n".join(parts).strip()

    if isinstance(result, str):
        return result.strip()

    if result is not None:
        return json.dumps(result, ensure_ascii=False)

    # Fallbacks si le format change.
    for key in ("result", "text", "content", "message"):
        if key in detail:
            return content_to_text(detail[key]).strip()

    # Dernier recours : on log et on renvoie le JSON brut visible.
    print(
        "[WARN] Format de reponse 1min.ai inattendu :\n"
        + json.dumps(response_json, ensure_ascii=False, indent=2),
        file=sys.stderr,
    )
    return json.dumps(response_json, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Appel API 1min.ai
# --------------------------------------------------------------------------- #

def call_1min_api(prompt: str, model: str, timeout_seconds: int) -> str:
    api_key = os.getenv("ONE_MIN_API_KEY")
    if not api_key:
        raise RuntimeError("ONE_MIN_API_KEY manquante. Exporte-la avant de lancer le proxy.")

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
        "type": os.getenv("ONE_MIN_FEATURE_TYPE", "CODE_GENERATOR"),
        "model": model,
        "promptObject": prompt_object,
    }

    if env_bool("ONE_MIN_LOG_PROMPTS", False):
        print("----- PROMPT -----\n" + prompt + "\n----- FIN -----", file=sys.stderr)

    request = urllib.request.Request(
        ONE_MIN_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "API-KEY": api_key,
            "User-Agent": os.getenv(
                "ONE_MIN_USER_AGENT",
                "Mozilla/5.0 (compatible; OneMinAiderProxy/2.0)",
            ),
            "Connection": "close",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Erreur HTTP 1min.ai {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Erreur reseau vers 1min.ai: {exc}") from exc

    if env_bool("ONE_MIN_LOG_RAW", False):
        print("----- RAW 1MIN.AI -----\n" + raw + "\n----- FIN RAW -----", file=sys.stderr)

    parsed = json.loads(raw)
    return sanitize_output(extract_1min_text(parsed))


# --------------------------------------------------------------------------- #
# Construction des reponses OpenAI
# --------------------------------------------------------------------------- #

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
    tokens = max(1, len(content.split()))
    return {
        "id": f"chatcmpl-1min-{now}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": tokens,
            "total_tokens": tokens,
        },
    }


def responses_response(model: str, content: str) -> Dict[str, Any]:
    now = int(time.time())
    rid = f"resp_1min_{now}"
    tokens = max(1, len(content.split()))
    return {
        "id": rid,
        "object": "response",
        "created_at": now,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": [{
            "id": f"msg_1min_{now}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": content,
                "annotations": [],
            }],
        }],
        "output_text": content,
        "parallel_tool_calls": False,
        "previous_response_id": None,
        "reasoning": None,
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "none",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 0,
            "output_tokens": tokens,
            "total_tokens": tokens,
        },
        "user": None,
        "metadata": {},
    }


# --------------------------------------------------------------------------- #
# Streaming SSE
# --------------------------------------------------------------------------- #

def split_chunks(text: str, size: int) -> List[str]:
    if size <= 0:
        return [text]
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def stream_chat_completion(handler: BaseHTTPRequestHandler, model: str, content: str) -> None:
    now = int(time.time())
    cid = f"chatcmpl-1min-{now}"
    chunk_size = env_int("ONE_MIN_STREAM_CHUNK") or 40

    def emit(delta: Dict[str, Any], finish: Optional[str]) -> None:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        handler.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8"))
        handler.wfile.flush()

    emit({"role": "assistant"}, None)
    for piece in split_chunks(content, chunk_size):
        emit({"content": piece}, None)
    emit({}, "stop")

    handler.wfile.write(b"data: [DONE]\n\n")
    handler.wfile.flush()


def sse_event(handler: BaseHTTPRequestHandler, event: str, data: Dict[str, Any]) -> None:
    handler.wfile.write(f"event: {event}\n".encode("utf-8"))
    handler.wfile.write(("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode("utf-8"))
    handler.wfile.flush()


def stream_responses(handler: BaseHTTPRequestHandler, model: str, content: str) -> None:
    now = int(time.time())
    rid = f"resp_1min_{now}"
    item_id = f"msg_1min_{now}"
    chunk_size = env_int("ONE_MIN_STREAM_CHUNK") or 40

    created = responses_response(model, "")
    created["id"] = rid
    created["status"] = "in_progress"
    created["output"] = []
    created["output_text"] = ""
    sse_event(handler, "response.created", {"type": "response.created", "response": created})

    sse_event(handler, "response.output_item.added", {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"id": item_id, "type": "message", "status": "in_progress",
                 "role": "assistant", "content": []},
    })

    sse_event(handler, "response.content_part.added", {
        "type": "response.content_part.added",
        "item_id": item_id, "output_index": 0, "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    })

    for piece in split_chunks(content, chunk_size):
        sse_event(handler, "response.output_text.delta", {
            "type": "response.output_text.delta",
            "item_id": item_id, "output_index": 0, "content_index": 0,
            "delta": piece,
        })

    sse_event(handler, "response.output_text.done", {
        "type": "response.output_text.done",
        "item_id": item_id, "output_index": 0, "content_index": 0,
        "text": content,
    })

    sse_event(handler, "response.content_part.done", {
        "type": "response.content_part.done",
        "item_id": item_id, "output_index": 0, "content_index": 0,
        "part": {"type": "output_text", "text": content, "annotations": []},
    })

    sse_event(handler, "response.output_item.done", {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {"id": item_id, "type": "message", "status": "completed",
                 "role": "assistant",
                 "content": [{"type": "output_text", "text": content, "annotations": []}]},
    })

    final = responses_response(model, content)
    final["id"] = rid
    sse_event(handler, "response.completed", {"type": "response.completed", "response": final})

    handler.wfile.write(b"data: [DONE]\n\n")
    handler.wfile.flush()


# --------------------------------------------------------------------------- #
# HTTP Handler
# --------------------------------------------------------------------------- #

class OneMinAiderProxyHandler(BaseHTTPRequestHandler):
    server_version = "OneMinAiderProxy/2.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def send_json(self, status: int, body: Dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)
        self.close_connection = True

    def send_sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json(status, openai_error_body(message, status))

    def read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        return json.loads(raw) if raw else {}

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]

        if path in {"/", "/health", "/v1/health"}:
            self.send_json(200, {
                "status": "ok",
                "service": "one_min_ai_aider_proxy",
                "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/responses"],
            })
            return

        if path in {"/v1/models", "/models"}:
            default_model = os.getenv("ONE_MIN_MODEL", "gpt-4o")
            self.send_json(200, {
                "object": "list",
                "data": [{
                    "id": default_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "1min.ai",
                }],
            })
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
                self.send_error_json(400, "'messages' doit etre une liste.")
                return

            model = normalize_model(body.get("model"))
            stream = bool(body.get("stream", False))
            timeout_seconds = int(os.getenv("ONE_MIN_TIMEOUT", "180"))

            prompt = chat_messages_to_prompt(messages)
            result = call_1min_api(prompt, model, timeout_seconds)

            if stream:
                self.send_sse_headers()
                stream_chat_completion(self, model, result)
                self.close_connection = True
                return

            self.send_json(200, chat_completion_response(model, result))

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

            prompt = responses_input_to_prompt(body)
            result = call_1min_api(prompt, model, timeout_seconds)

            if stream:
                self.send_sse_headers()
                stream_responses(self, model, result)
                self.close_connection = True
                return

            self.send_json(200, responses_response(model, result))

        except json.JSONDecodeError as exc:
            self.send_error_json(400, f"JSON invalide: {exc}")
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(502, str(exc))


# --------------------------------------------------------------------------- #
# Entree
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proxy OpenAI-compatible pour Aider via 1min.ai."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'ecoute.")
    parser.add_argument("--port", type=int, default=8787, help="Port d'ecoute.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), OneMinAiderProxyHandler)

    print(f"Proxy 1min.ai demarre sur http://{args.host}:{args.port}/v1", flush=True)
    print(f"Feature type : {os.getenv('ONE_MIN_FEATURE_TYPE', 'CODE_GENERATOR')}", flush=True)
    print("Endpoints: GET /v1/models, POST /v1/chat/completions, POST /v1/responses", flush=True)

    if not os.getenv("ONE_MIN_API_KEY"):
        print("Attention: ONE_MIN_API_KEY non defini.", file=sys.stderr, flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArret du proxy.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
