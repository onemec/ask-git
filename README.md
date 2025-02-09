# ask-git
Ingest a Git repo and query the data with an LLM

## Setup:

1. Download and install [Ollama](https://ollama.com/) and [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. `ollama pull deepseek-r1:8b`
3. `ollama serve`
4. Open another terminal in the root of this Git repo
5. `uv venv`
6. `uv run main.py pull`
7. `uv run main.py ask`
