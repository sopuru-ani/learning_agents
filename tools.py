import platform
import re
from datetime import datetime
from pathlib import Path

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool

from shell import close_shell, get_shell

search_tool = DuckDuckGoSearchRun()

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

SENSITIVE_COMMAND_PATTERNS = [
    re.compile(r"\bgit\s+push\b", re.I),
    re.compile(r"\bgit\s+pull\b", re.I),
    re.compile(r"\bgit\s+fetch\b", re.I),
    re.compile(r"\brm\b", re.I),
    re.compile(r"\bmv\b", re.I),
    re.compile(r"\bsudo\b", re.I),
    re.compile(r"\b(chmod|chown)\b", re.I),
    re.compile(r"\b(pip|pip3|npm|apt|dnf|yum|brew)\s+(install|uninstall|remove)\b", re.I),
    re.compile(r"\bgit\s+config\b", re.I),
    re.compile(r"(curl|wget).*\|\s*(ba)?sh", re.I),
]


def _is_sensitive_command(command: str) -> bool:
    return any(pattern.search(command) for pattern in SENSITIVE_COMMAND_PATTERNS)


def _confirm_sensitive_command(command: str) -> bool:
    print(f"\n[confirm] Sensitive command blocked until you approve:")
    print(f"  {command}")
    answer = input("Type 'yes' to run, anything else to cancel: ").strip().lower()
    return answer == "yes"


@tool
def save_text_to_file(data: str, filename: str = "research_output.txt") -> str:
    """Persist text to a file on disk. Use when the user wants results saved.
    Pass the full research content as data and a descriptive filename."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filepath}"

def _format_shell_result(command: str, output: str, exit_code: int) -> str:
    parts = [
        f"shell: persistent system shell ({platform.system()}, project venv not active)",
        f"exit_code: {exit_code}",
    ]
    if output:
        parts.append(f"output:\n{output.rstrip()}")
    else:
        parts.append("(no output)")
    parts.append(f"command:\n{command}")
    return "\n\n".join(parts)


@tool
def terminal_run(command: str) -> str:
    """Run a command in a persistent system shell and return combined output and exit code.

    The shell keeps state across calls (e.g. cd persists). It runs outside the
    project's Python venv so system tools and packages are visible. stdout and
    stderr are merged in the output.

    Sensitive commands (git push, rm, sudo, package installs, etc.) are blocked
    until the user types 'yes' at an interactive prompt."""
    if _is_sensitive_command(command) and not _confirm_sensitive_command(command):
        return (
            "exit_code: blocked\n\n"
            "Command not run: user denied permission.\n\n"
            f"command:\n{command}"
        )

    try:
        output, exit_code = get_shell().run(command)
    except RuntimeError as exc:
        return f"exit_code: error\n\n{exc}\n\ncommand:\n{command}"

    return _format_shell_result(command, output, exit_code)

@tool
def file_read(filepath: str) -> str:
    """Read the contents of a file and return the output. This uses the Pathlib library to read the file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

@tool
def file_write(filepath: str, content: str) -> str:
    """Write to a file and return the output. This uses the Pathlib library to write to the file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File {filepath} successfully written."