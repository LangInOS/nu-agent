import os
import re
import subprocess
from pathlib import Path
from typing import List

from anthropic import Anthropic
from anthropic.types import ContentBlock, Message, MessageParam, ToolParam
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd() / "workspace"
client: Anthropic = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL: str = os.getenv("MODEL_ID", "glm-5")

SKILLS_DIR = WORKDIR / "skills"


# === SECTION: basic tools ===
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


# === SECTION: todos manager ===
class TodoManager:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")

        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))

            if not text:
                raise ValueError(f"Item {item_id}: text required")

            if status not in ["pending", "in_progress", "completed"]:
                raise ValueError(f"Item {item_id}: invalid status '{status}'")

            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[  ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

# === SECTION: skills loader ===
class SkillLoader:
    def __init__(self, skills_path: Path) -> None:
        self.skills = {}
        if skills_path.exists():
            for f in sorted(skills_path.rglob("SKILL.md")):
                text = f.read_text()
                match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
                meta, body = {}, text
                if match:
                    for line in match.group(1).strip().splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            meta[k.strip()] = v.strip()
                    body = match.group(2).strip()
                name = meta.get("name", f.parent.name)
                self.skills[name] = {"meta": meta, "body": body}

    def descriptions(self) -> str:
        if not self.skills: return "(no skills)"
        return "\n".join(f" - {n}: {s['meta'].get('description', '-')}" for n, s in self.skills.items())

    def load(self, name: str) -> str:
        s = self.skills.get(name)
        if not s: return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{s['body']}\n</skill>"

# === SECTION: global instances ===
TODO = TodoManager()
SKILLS = SkillLoader(SKILLS_DIR)


# === SECTION: system prompt ===
SYSTEM = f"""
You are a coding agent at {WORKDIR}. Use tools to solve tasks.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Use the task tool to delegate exploration or subtasks.
Use load_skill for specialized knowledge before tacking unfamiliar topics.
Skills available: {SKILLS.descriptions()}
"""

# === SECTION: tool dispatch===
TOOLS: list[ToolParam] = [
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
    {
        "name": "todo",
        "description": "Update task list. Track progress on multi-step tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["id", "text", "status"],
                    },
                }
            },
            "required": ["items"],
        },
    },
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
    },
    {
        "name": "load_skill",
        "description": "Load specialized knowledge by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "skill name to load"}
            },
            "required": ["name"]
        }
    }
]


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"],
                                       kw["new_text"]),
    "todo": lambda **kw: TODO.update(kw["items"]),
    "task": lambda **kw: run_subagent(kw["prompt"]),
    "load_skill": lambda **kw: SKILLS.load(kw["name"])
}

# Subagent
def run_subagent(prompt: str) -> str:
    sub_tool_names = {"bash", "read_file", "write_file", "edit_file"}
    sub_tools = [t for t in TOOLS if t["name"] in sub_tool_names]
    sub_handlers = {k: v for k, v in TOOL_HANDLERS.items() if k in sub_tool_names}

    SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."
    sub_msgs: list[MessageParam] = [{"role": "user", "content": prompt}]
    sub_resp = None
    for _ in range(30):
        sub_resp = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM, messages=sub_msgs,tools=sub_tools, max_tokens=8192)
        sub_msgs.append({"role": "assistant", "content": sub_resp.content})
        if sub_resp.stop_reason != "tool_use":
            break
        results = []
        for block in sub_resp.content:
            if block.type == "tool_use":
                handler = sub_handlers.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                results.append({
                    "type": "tool_result", 
                    "tool_use_id": block.id, 
                    "content": str(output)[:50000]})
        sub_msgs.append({"role": "user", "content": results})

    if sub_resp:
        return "".join(b.text for b in sub_resp.content if hasattr(b, "text")) or "(no summary)"

    return "(subagent failed)"


# Agent Loop
def agent_loop(messages: list[MessageParam]) -> List[ContentBlock]:
    rounds_since_todo = 0
    while True:
        print(f"\033[36mthinking...\n\033[0m")
        response: Message = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages, max_tokens=8192, tools=TOOLS
        )

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response.content

        results: list = []
        used_todo = False
        for block in response.content:
            if block.type == "text":
                print(f"\033[90m[Model Thought]: {block.text}\033[0m")

            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                print("\033[32m--------------------\033[0m")
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"\033[32m> {block.name}: {output}\033[0m")
                print("\033[32m--------------------\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
                if block.name == "todo":
                    used_todo = True
                    print("\n\033[33m=== 📋 INITIAL PLAN (Todo List) ===\033[0m")
                    print(output)
                    print("\033[33m=================================\033[0m\n")
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            results.insert(0, {"type": "text", "text": "<reminder>Update your todos.</reminder>"})

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

        print("\n final reponse: \n")

        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(f"\033[35m{block.text}\033[0m")
                else:
                    print(f"\033[35m{block}\033[0m")

        print()
