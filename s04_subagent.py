import os
import subprocess
from pathlib import Path
from typing import List

from anthropic import Anthropic
from anthropic.types import ContentBlock, Message, MessageParam, ToolParam
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()

client: Anthropic = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

MODEL: str = os.getenv("MODEL_ID", "glm-5")


SYSTEM = f"""You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."""
SUBAGENT_SYSTEM = f"You are a coding subagent at{WORKDIR}. Complete the given task, then summarize your findings."


# Basic Tools
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    forbidden = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in forbidden):
        return "Error: Dangerous command blocked"

    try:
        result = subprocess.run(
            command, shell=True, cwd=WORKDIR, text=True, timeout=120, capture_output=True
        )

        out = (result.stdout + result.stderr).strip()

        return out[:50000] if out else "(no output)"
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


BAISC_TOOLS: list[ToolParam] = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
]

TOOLS = BAISC_TOOLS + [
    {
        "name": "task",
        "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string"
                },
                "description": {
                    "type": "string",
                    "description": "Short description of the task"
                }
            },
            "required": ["prompt"]
        }
    }
]

TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"],
                                       kw["new_text"]),
}

# Subagent
def run_subagent(prompt: str) -> str:
    sub_messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    response = None
    for _ in range(30):
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM, messages=sub_messages,
            tools=BAISC_TOOLS, max_tokens=8192
        )
        sub_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)[:50000]})
        sub_messages.append({"role": "user", "content": results})

    if not response:
        return "(no summary)"
    else:
        return "".join(b.text for b in response.content if hasattr(b, "text")) or "(no summary)"



# Agent Loop
def agent_loop(messages: list[MessageParam]) -> List[ContentBlock]:
    while True:
        print(f"\033[36mthinking...\n\033[0m")
        response: Message = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages, max_tokens=8192, tools=TOOLS
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return response.content

        results: list = []
        for block in response.content:
            if block.type == "text":
                print(f"\033[90m[Model Thought]: {block.text}\033[0m")

            if block.type == "tool_use":
                if block.name == "task":
                    desc = block.input.get("description", "subagent")
                    prompt = str(block.input["prompt"])
                    print(f"> task({desc}): {prompt}")
                    output = run_subagent(prompt=prompt)
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    except Exception as e:
                        output = f"Error: {e}"
                print("\033[32m--------------------\033[0m")
                print(f"\033[32m> {block.name}: {output}\033[0m")
                print("\033[32m--------------------\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history: list[MessageParam] = []
    while True:
        try:
            query: str = input("\033[36mYou >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})

        response_content: List[ContentBlock] = agent_loop(history)

        print("\n")

        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    text = block.text
                    print(f"\033[35m{text}\033[0m")
                else:
                    print(f"\033[35m{block}\033[0m")

        print()

