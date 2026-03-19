
import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import Message, MessageParam, ToolParam, ContentBlock
from typing import List

from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()

client: Anthropic = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

MODEL: str | None = os.getenv("MODEL_ID")

if not MODEL:
    raise ValueError("MODEL_ID environment variable not set")

SYSTEM: str = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."

TOOLS: list[ToolParam] = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": { "command": { "type": "string" } },
            "required": ["command"]

        }
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": { "type": "string"  },
                "limit": { "type": "integer" }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "old_text": { "type": "string" },
                "new_text": { "type": "string"  }
            },
            "required": ["path", "old_text", "new_text"]
        }
    },
]

TOOL_HANDLERS = {
    "bash":         lambda **kw: run_bash(kw["command"]),
    "read_file":    lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":   lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":    lambda **kw: run_edit(kw["path"], 
                                          kw["old_text"],
                                          kw["new_text"]),
}

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str):
    forbidden = ["rm -fr /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in forbidden):
        return "Error: Dangerous command blocked"

    try:
        result = subprocess.run(command,
                                shell=True,
                                cwd=WORKDIR,
                                text=True,
                                timeout=120,
                                capture_output=True)

        out = (result.stdout + result.stderr).strip()

        return out if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int | None = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"



# -- agent loop: calls tools until the llm stops --
def agent_loop(messages: list[MessageParam]) -> List[ContentBlock]:
    while(True):
        print(f"\033[36mthinking...\n\033[0m")
        response: Message = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            max_tokens=8192,
            tools=TOOLS
        )

        messages.append({ "role": "assistant", "content": response.content })

        if response.stop_reason != "tool_use":
            return response.content

        results: list[MessageParam] = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                print("\033[32m--------------------\033[0m")
                output = handler(**block.input) if handler else f"Unknow tool: {block.name}"
                print(f"\033[32m> {block.name}: {output}\033[0m")
                print("\033[32m--------------------\033[0m")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output }
                )

        messages.append({ "role": "user", "content": results })



if __name__ == "__main__":
    history: list[MessageParam] = []
    while(True):
        try:
            query: str = input("\033[36mYou >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({ "role" : "user", "content": query })

        response_content: List[ContentBlock] = agent_loop(history)

        print("\n")

        if isinstance(response_content, List):
            for block in response_content:
                if hasattr(block, "text"):
                    print(f"\033[35m{block.text}\033[0m")
                else:
                    print(f"\033[35m{block}\033[0m")

        print()

