import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from shell import close_shell
from tools import save_text_to_file, search_tool, wikipedia_tool, terminal_run, file_read, file_write

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    filename: str


SYSTEM_PROMPT = """
/no_think
You are a local system assistant that helps with tasks on the system and occasionally research tasks.

Environment:
- You (this Python app) run inside the project's virtual environment with LangChain dependencies.
- terminal_run uses a separate persistent system shell without that venv on PATH.
- The shell keeps state between calls: cd, exported variables, and similar changes persist.
- Do not assume fixed paths (/usr/bin, etc.). Discover tools on this machine at runtime.
- Discover the OS before assuming command syntax:
  - Linux/macOS: uname -s, pwd, command -v <tool>, which -a python3
  - Windows: ver, cd, where <tool>
- To compare Python installs: python3 -c "import sys; print(sys.executable, sys.prefix)"
- Never hardcode OS-specific paths; verify on the current system first.

Work in steps:
1. If you need external facts, call the research tools available to you.
2. Read tool results in the conversation before deciding your next step.
3. If the user wants output persisted, call the appropriate tool to write results
   to disk before you finish.
4. If the user wants to read a file, call the appropriate tool to read the file.
5. If the user wants to write to a file, call the appropriate tool to write to the file.
6. If the user wants to run a terminal command, call terminal_run — never guess what
   a command would output.
7. Base answers about the system, files, git, or auth only on actual tool results.

Sensitive commands — ask before running:
Before calling terminal_run for a sensitive command, stop and ask the user for explicit
permission in chat. State the exact command you want to run and wait for approval.
terminal_run also enforces this: sensitive commands block until the user types 'yes'
at the terminal prompt. If they cancel, report that the command was not run.

Sensitive commands include (not exhaustive):
- git push, git pull, or any command that changes a remote
- rm, mv, or other destructive file operations
- sudo or commands run as another user
- installing or uninstalling packages (pip, npm, apt, etc.)
- modifying credentials, SSH keys, git config, or .env files
- curl/wget piped to a shell, or downloading and executing scripts
- any command the user did not ask for that writes outside the project directory

Safe to run without asking when needed to answer the user:
- read-only inspection and OS discovery (ls/dir, cat/type, pwd/cd, git status, etc.)
- auth/connectivity checks the user requested: ssh -T git@github.com, gh auth status
- other non-destructive diagnostics clearly tied to the user's question

Choose tools from their descriptions and the user request. Do not invent tool usage.
"""

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_ENDPOINT"),
    max_tokens=4096,
    temperature=0.6,
)

agent = create_agent(
    model=llm,
    tools=[search_tool, wikipedia_tool, save_text_to_file, terminal_run, file_read, file_write],
    system_prompt=SYSTEM_PROMPT,
    debug=True,
    middleware=[
        ToolRetryMiddleware(
            max_retries=2,
            on_failure="continue",
        ),
    ],
)

messages = []

while True:
    query = input("~>: ")
    if query.strip() == "/bye":
        close_shell()
        break

    prev_count = len(messages)
    messages.append(HumanMessage(content=query))
    result = agent.invoke({"messages": messages})
    messages = result["messages"]

    for msg in messages[prev_count:]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"[tool] {tc['name']}({tc['args']})")

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            print(msg.content)
            break
