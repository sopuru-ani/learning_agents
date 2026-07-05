# learning_agents

A local CLI assistant for system tasks, codebase work, and light research. Run it from any project folder — it binds to that directory automatically.

## Quick start

```bash
cd ~/Code/my-project
python ~/Code/learning_agents/main.py
```

Optional shell alias (add to `~/.zshrc`):

```bash
alias local-agent='~/Code/learning_agents/venv/bin/python ~/Code/learning_agents/main.py'
```

Then: `cd` into a project and run `local-agent`.

## Setup

```bash
cd ~/Code/learning_agents
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in this directory — see **Configuration** below. Requires Python 3.10+.

## REPL commands

| Command | Description |
|---------|-------------|
| `/shell` | Interactive subshell (`/chat` to return) |
| `/shell <cmd>` | Run one command in a fresh subshell |
| `/clear` | Wipe this project's chat session |
| `/bye` | Exit |

Chat input at `~>` does **not** run shell commands — the agent must call `terminal_run`, or you use `/shell`.

Set `AGENT_DEBUG=1` for raw LangChain logs.

## Memory

Three tiers, keyed automatically from the directory you launch in:

| Tier | Storage | Purpose |
|------|---------|---------|
| **Session** | `.agent/session.sqlite` | Full chat history for this project |
| **Project (Cognee)** | `.agent/cognee/` | Durable facts about this repo |
| **Device (Cognee)** | `~/.config/learning_agents/cognee/` | Machine-wide facts (OS, tools, preferences) |

- One session per project folder; survives restart until `/clear`.
- Cognee tools: `remember_project`, `remember_device`, `recall_project`, `recall_device`.
- Add `.agent/` to your project's `.gitignore` (already ignored in this repo).

### Testing Cognee

On startup, look for `Cognee: ready`. Then try:

```
remember that this machine uses Fedora and dnf
```

Quit with `/bye`, restart, and ask about your package manager. If it recalls the fact, device memory is working.

## Tools

| Tool | Use |
|------|-----|
| `terminal_run` | Non-interactive shell commands (persistent shell, venv stripped from PATH) |
| `file_read` / `file_write` | Read or write project files |
| `search` / `wikipedia` | External lookup when needed |
| `remember_*` / `recall_*` | Long-term Cognee memory |

Interactive commands (`gh auth login`, `ssh`, editors, REPLs) are blocked — use `/shell` instead. Sensitive commands (`git push`, `rm`, `sudo`, package installs) require typing `yes` at a prompt.

## Configuration (`.env`)

Chat and Cognee use **separate** LLM settings so you can keep direct NVIDIA model ids for chat while Cognee routes through LiteLLM.

**Chat agent** (`LLM_*`) — used by LangChain, talks directly to `LLM_ENDPOINT`:

```dotenv
LLM_MODEL="nvidia/your-model-id"
LLM_ENDPOINT="https://integrate.api.nvidia.com/v1"
LLM_API_KEY="nvapi-..."
```

**Cognee** (`COGNEE_LLM_*`, `COGNEE_EMBEDDING_*`) — used only for remember/recall:

```dotenv
COGNEE_LLM_PROVIDER="custom"
COGNEE_LLM_MODEL="hosted_vllm/nvidia/your-model-id"
COGNEE_LLM_ENDPOINT="https://integrate.api.nvidia.com/v1"
COGNEE_LLM_API_KEY="nvapi-..."

COGNEE_EMBEDDING_PROVIDER="openai_compatible"
COGNEE_EMBEDDING_MODEL="nvidia/nv-embed-v1"
COGNEE_EMBEDDING_ENDPOINT="https://integrate.api.nvidia.com/v1"
COGNEE_EMBEDDING_API_KEY="nvapi-..."
COGNEE_EMBEDDING_DIMENSIONS="4096"
```

The `hosted_vllm/` prefix on `COGNEE_LLM_MODEL` tells LiteLLM to use your endpoint. Embeddings use `openai_compatible` to avoid LiteLLM quirks with NVIDIA.

Commented blocks in `.env` show how to swap to **Ollama** for local chat and/or Cognee — uncomment the matching LLM and embedding sections together.

## Project structure

```
.
├── main.py              # REPL, agent setup, system prompt
├── tools.py             # LangChain tools (shell, files, search, memory)
├── shell.py             # Persistent piped bash for the agent
├── ui.py                # Terminal formatting
├── agent_paths.py       # Project root detection, .agent/ and device paths
├── session_memory.py    # LangGraph SQLite session persistence
├── cognee_memory.py     # Cognee project/device tiers
├── requirements.txt
├── .env                 # LLM and Cognee config (not committed)
└── .agent/              # Per-project data (session + Cognee), gitignored
```

## Dependencies

LangChain, LangGraph (SQLite checkpointer), Cognee, DuckDuckGo search, Wikipedia, python-dotenv, pydantic. LLM backend is any OpenAI-compatible API (NVIDIA NIM, Ollama, etc.) configured via `.env`.
