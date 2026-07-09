# Guide rapide d'installation d'Aider avec 1min.ai

## 1. Créez un environnement virtuel Python 3.12
```bash
py -3.12 -m venv aider_env
# ou
python3.12 -m venv aider_env
```

## 2. Activez l'environnement
```bash
# PowerShell
.\aider_env\Scripts\activate

# Git Bash, macOS ou WSL
source aider_env/Scripts/activate
# ou
source aider_env/bin/activate
```

## 3. Mettez pip à jour
```bash
python3.12 -m pip install --upgrade pip
```

## 4. Installez aider-chat
```bash
python3.12 -m pip install aider-chat
```

## 5. Lancez le proxy OneMin
Ouvrez un terminal (Git Bash ou équivalent) et exécutez :
```bash
cd
export ONE_MIN_API_KEY=xxx
export ONE_MIN_MODEL=gpt-5.3-codex
python3.12 one_min_ai_aider_proxy.py --port 8787 &
```

## 6. Lancez Aider via le proxy
Dans un autre terminal :
```bash
cd
export ONE_MIN_MODEL=gpt-5.3-codex
winpty aider --model openai/gpt-5.3-codex --weak-model openai/gpt-5.1-codex-mini --openai-api-base http://127.0.0.1:8787/v1 --openai-api-key dummy --no-auto-commits --map-tokens 8192 --yes  --no-stream
```

### Options supplémentaires
```bash
# winpty aider --model $ONE_MIN_MODEL --openai-api-base http://127.0.0.1:8787/v1 --openai-api-key dummy --no-auto-commits
# winpty aider --openai-api-base http://127.0.0.1:8787/v1 --openai-api-key dummy --no-auto-commits
# winpty aider --model openai/gpt-4o --openai-api-base http://127.0.0.1:8787/v1 --openai-api-key dummy --no-auto-commits
# Avec --edit-format whole, Aider demande au modèle de réécrire le fichier complet, ce qui évite les erreurs de correspondance exacte dans SEARCH.
```
