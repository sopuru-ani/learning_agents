import os
import subprocess

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from shell import close_shell, system_shell_env
from tools import save_text_to_file, search_tool, wikipedia_tool, terminal_run, file_read, file_write
from ui import agent_calling_tool, agent_reply_start

load_dotenv()

AGENT_DEBUG = os.getenv("AGENT_DEBUG", "").lower() in ("1", "true", "yes")


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

Interactive commands — never run via terminal_run:
terminal_run cannot handle programs that need keyboard input (login wizards, ssh sessions,
editors, REPLs). It will refuse them with exit_code: blocked.
When the user needs one (e.g. gh auth login), tell them clearly to run it in Konsole or
their system terminal, then continue after they confirm.

Never pretend a command ran:
- If the user pastes a shell command in chat, that does NOT execute it. Call terminal_run
  for non-interactive commands, or direct them to Konsole for interactive ones.
- Never say a command is "running" or "initiated" unless terminal_run was called and you
  have its tool result in the conversation.

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
    debug=AGENT_DEBUG,
    middleware=[
        ToolRetryMiddleware(
            max_retries=2,
            on_failure="continue",
        ),
    ],
)

messages = []


def _print_stream_updates(event: dict) -> None:
    for update in event.values():
        if not isinstance(update, dict):
            continue
        for msg in update.get("messages", []):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    agent_calling_tool(tc["name"], tc.get("args", {}))


def _latest_reply(new_messages: list) -> str | None:
    for msg in reversed(new_messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return None


print("Tool and shell activity print as they run. Set AGENT_DEBUG=1 for raw LangChain logs.\n")
print("Commands: /shell, /shell <cmd>, /chat (in shell mode), /clear, /bye\n")
while True:
    query = input("~>: ")
    if query.strip() == "/bye":
        close_shell()
        break
    
    if query.strip() == "/clear":
        messages = []
        print("Messages cleared.")
        continue
    
    if query.strip() == "/shell":
        print("Shell mode - type /chat to return to chat mode. Each line runs in a fresh subshell")
        while True:
            line = input("(shell)$")
            if line.strip() == "/chat":
                break
            if line.strip():
                subprocess.run(line, shell=True, env=system_shell_env())
        continue
    
    if query.startswith("/shell "):
        command = query[len("/shell "):].strip()
        if not command:
            print("Usage: /shell <command>")
            continue
        subprocess.run(command, shell=True, env=system_shell_env())
        continue


    prev_count = len(messages)
    messages.append(HumanMessage(content=query))

    final_state = None
    for chunk in agent.stream(
        {"messages": messages},
        stream_mode=["updates", "values"],
    ):
        mode, payload = chunk
        if mode == "updates":
            _print_stream_updates(payload)
        elif mode == "values":
            final_state = payload

    if final_state is None:
        print("Error: agent produced no response.")
        messages.pop()
        continue

    messages = final_state["messages"]
    reply = _latest_reply(messages[prev_count:])
    if reply:
        agent_reply_start()
        print(reply)
