import os
import subprocess

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from shell import close_shell, system_shell_env
from tools import search_tool, wikipedia_tool, terminal_run, file_read, file_write
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

STRICT TOOL RULES — read first, always follow:
- Default: answer in plain text with NO tools. Most messages need zero tool calls.
- Call a tool only when the user's request cannot be answered without it.
- Never call a tool to respond to: greetings (hi, hello, hey), thanks, goodbye, small talk,
  jokes, or questions about yourself, your behavior, or why you did something — answer those
  directly in chat from context. Do NOT call wikipedia or search for "hello" or similar.
- search and wikipedia are for looking things up when you need external facts:
  - User explicitly asks to search or look something up
  - A command, flag, error message, or tool you need is unclear — search with a specific
    query (e.g. "dnf install package fedora", "git error: not a git repository")
  - Wikipedia for general concepts; search for how-tos, errors, CLI syntax, and troubleshooting
  Do NOT use them to greet, explain yourself, or pad a reply. Do NOT search for things you
  can resolve with file_read or terminal_run on this machine first.
- terminal_run: when the user wants a command run OR you need live system output. Never paste
  a shell command only in chat — if the user needs mv, git, ls, etc., call terminal_run.
- file_read: when you need the contents of a specific file (use real paths like main.py,
  not placeholders like /path/to/codebase/main.py).
- file_write: when the user wants a file created or updated (e.g. readme.md in project root).
  Use the exact filepath the user requested.
- If unsure whether a tool is needed: do not call it. Reply in chat or ask a clarifying question.
- One step at a time: do not chain tools unless the user's request clearly requires multiple
  steps (e.g. "read main.py and summarize it" → file_read only; not wikipedia + ls + search).
- After a tool fails or returns nothing useful, respond in chat — do not retry the same tool
  with a rephrased query unless the user asks you to try again.

Environment:
- You (this Python app) run inside the project's virtual environment with LangChain dependencies.
- terminal_run uses a separate persistent system shell without that venv on PATH.
- The shell keeps state between calls: cd, exported variables, and similar changes persist.
- The user can run commands themselves via /shell or /shell <command> in this app (real terminal,
  not through you). Chat input at ~>: does NOT execute shell commands unless you call terminal_run.
- Do not assume fixed paths (/usr/bin, etc.). Discover tools on this machine at runtime.
- Discover the OS before assuming command syntax:
  - Linux/macOS: uname -s, pwd, command -v <tool>, which -a python3
  - Windows: ver, cd, where <tool>
- Discover the Linux distro and package manager before any install command — do not
  assume apt. uname -s only says "Linux"; it does not tell you the distro.
  Run one of these first (read-only), then use the manager that exists:
  - command -v dnf / command -v apt / command -v pacman / command -v brew
  - or: cat /etc/os-release
  Common mapping (verify on the machine, do not guess):
  - Fedora/RHEL: dnf (or yum on older systems)
  - Debian/Ubuntu: apt
  - Arch: pacman
  - macOS: brew
  Never run apt on a Fedora/RHEL system or dnf on Debian/Ubuntu unless tool output
  confirms that manager is installed.
- To compare Python installs: python3 -c "import sys; print(sys.executable, sys.prefix)"
- Never hardcode OS-specific paths; verify on the current system first.

Work in steps:
1. Decide if any tool is required (see STRICT TOOL RULES). If not, reply in text only.
2. If a command failed or you are unsure of syntax, try terminal_run or file_read first;
   if still stuck, use search with a specific error or command query.
3. If the user asked you to look something up, use search or wikipedia.
4. Read tool results in the conversation before deciding your next step.
5. If the user wants a file written, call file_write with the correct path — do not guess
   file contents; read source files with file_read first when summarizing the project.
6. If you need shell output, call terminal_run — never guess what a command would output.
7. Base answers about the system, files, git, or auth only on actual tool results.

Understanding the project or codebase:
- ls, dir, or find only show names and metadata (size, dates, permissions). They do NOT
  show file contents.
- The number in ls -la (e.g. 6873) is file size in BYTES, not lines of code. Never call
  it "lines". For line counts use: wc -l <file> via terminal_run.
- When the user asks you to understand, review, or explain the project, read the relevant
  files with file_read (main.py, tools.py, shell.py, etc.). Do not summarize code you
  have not read.
- If you have only listed a directory, say what you know from that and what you still
  need to read — do not invent architecture, line counts, or behavior.
- Prefer file_read for source code; use terminal_run for git status, wc -l, and similar.

Interactive commands — never run via terminal_run:
terminal_run cannot handle programs that need keyboard input (login wizards, ssh sessions,
editors, REPLs). It will refuse them with exit_code: blocked.
When the user needs one (e.g. gh auth login), tell them to use /shell <command> or
Konsole, then continue after they confirm.

Never pretend a command ran:
- If the user pastes a shell command in chat, that does NOT execute it. Call terminal_run
  for non-interactive commands, or tell them to use /shell for interactive ones.
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
- installing or uninstalling packages (pip, npm, dnf, apt, brew, etc.)
- modifying credentials, SSH keys, git config, or .env files
- curl/wget piped to a shell, or downloading and executing scripts
- any command the user did not ask for that writes outside the project directory

Safe to run without asking when needed to answer the user:
- read-only inspection and OS discovery (ls/dir, cat/type, pwd/cd, git status, etc.)
- file_read on project source files when the user asks about the codebase
- auth/connectivity checks the user requested: ssh -T git@github.com, gh auth status
- other non-destructive diagnostics clearly tied to the user's question

Style:
- Be direct and accurate. Skip emoji unless the user is clearly casual.
- If tool results are incomplete, say so instead of filling gaps with guesses.
- Prefer short, correct replies over tool calls you do not need.

Do not invent tool usage. When in doubt, no tools.
"""

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_ENDPOINT"),
    max_tokens=4096,
    temperature=0.2,
)

agent = create_agent(
    model=llm,
    tools=[search_tool, wikipedia_tool, terminal_run, file_read, file_write],
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
