winpty



cd

source aider\_env/Scripts/activate

export ONE\_MIN\_API\_KEY=xxx

export ONE\_MIN\_MODEL=gpt-5.1-codex-mini

python one\_min\_ai\_aider\_proxy.py --port 8787 \&



cd

source aider\_env/Scripts/activate



export ONE\_MIN\_MODEL=gpt-5.1-codex-mini



winpty aider --model openai/gpt-5.1-codex-mini --weak-model openai/gpt-5.1-codex-mini --openai-api-base http://127.0.0.1:8787/v1 --openai-api-key dummy --no-auto-commits

\# winpty aider --model $ONE\_MIN\_MODEL --openai-api-base http://127.0.0.1:8787/v1 --openai-api-key dummy --no-auto-commits

\# winpty aider --openai-api-base http://127.0.0.1:8787/v1 --openai-api-key dummy --no-auto-commits

\# winpty aider --model openai/gpt-4o --openai-api-base http://127.0.0.1:8787/v1 --openai-api-key dummy --no-auto-commits



