import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from queue import Queue

from anthropic import Anthropic
from anthropic.types import ContentBlock, Message, MessageParam, ToolParam
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd() / "workspace"
client: Anthropic = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL: str = os.getenv("MODEL_ID", "glm-5")

TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
SKILLS_DIR = WORKDIR / "skills"
TASKS_DIR = WORKDIR / ".tasks"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3
TOKEN_THRESHOLD = 100000
POLL_INTERVAL = 5
IDLE_TIMEOUT = 60

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}


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
        result = subprocess.run(command, shell=True, cwd=WORKDIR, text=True, 
                                timeout=120, capture_output=True)
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


# === SECTION: todos ===
class TodoManager:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()
            activeForm = str(item.get("activeForm", "")).strip()
            item_id = str(item.get("id", str(i + 1)))
            if not content:
                raise ValueError(f"Item {item_id}: content required")
            if status not in ["pending", "in_progress", "completed"]:
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if not activeForm:
                raise ValueError(f"Item {item_id}: activeForm required")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "content": content, "status": status, 
                              "activeForm": activeForm})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(item["status"], "[?]")
            suffix = f" <- {item['activeForm']}" if item["status"] == "in_progress" else ""
            lines.append(f"{marker} #{item['id']}: {item['content']}{suffix}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

    def has_open_items(self) -> bool:
        return any(item.get("status") != "completed" for item in self.items)


# === SECTION: Subagent ===
def run_subagent(prompt: str, agent_type: str = "Explore") -> str:
    sub_tool_names = {"bash", "read_file"}
    if agent_type != "Explore":
        sub_tool_names.update({"write_file", "edit_file"})
    
    sub_tools = [t for t in TOOLS if t["name"] in sub_tool_names]
    sub_handlers = {k: v for k, v in TOOL_HANDLERS.items() if k in sub_tool_names}

    SUBAGENT_SYSTEM = f"""You are a coding subagent at {WORKDIR}. 
                        Complete the given task, then summarize your findings."""
    sub_msgs: list[MessageParam] = [{"role": "user", "content": prompt}]
    sub_resp = None
    for _ in range(30):
        sub_resp = client.messages.create(
            model=MODEL, 
            system=SUBAGENT_SYSTEM, 
            messages=sub_msgs, 
            tools=sub_tools, 
            max_tokens=8192
        )
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
                    "content": str(output)[:50000]
                })
        sub_msgs.append({"role": "user", "content": results})

    if sub_resp:
        return "".join(b.text for b in sub_resp.content if hasattr(b, "text")) or "(no summary)"

    return "(subagent failed)"


# === SECTION: skills ===
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
        if not self.skills:
            return "(no skills)"
        return "\n".join(f"  - {n}: {s['meta'].get('description', '-')}" 
                    for n, s in self.skills.items())

    def load(self, name: str) -> str:
        s = self.skills.get(name)
        if not s:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{s['body']}\n</skill>"


# === SECTION: tasks ===
class TaskManager:
    def __init__(self) -> None:
        TASKS_DIR.mkdir(exist_ok=True)

    def _next_id(self) -> int:
        ids = [int(f.stem.split("_")[1]) for f in TASKS_DIR.glob("task_*.json")]
        return max(ids, default=0) + 1

    def _load(self, task_id: int) -> dict:
        p = TASKS_DIR / f"task_{task_id}.json"
        if not p.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(p.read_text())

    def _save(self, task: dict) -> None:
        (TASKS_DIR / f"task_{task['id']}.json").write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        task = {"id": self._next_id(), "subject": subject, "description": description, 
                "status": "pending", "owner": None, "blockedBy": [], "blocks": []}
        self._save(task)
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2)

    def update(self, task_id: int, status: str = None, add_blocked_by: list = None, add_blocks: list = None) -> str:
        task = self._load(task_id)
        if status:
            task["status"] = status
            if status == "completed":
                for f in TASKS_DIR.glob(f"task_*.json"):
                    t = json.loads(f.read_text())
                    if task_id in t.get("blockedBy", []):
                        t["blockedBy"].remove(task_id)
                        self._save(t)
            if status == "deleted":
                (TASKS_DIR / f"task_{task_id}.json").unlink(missing_ok=True)
                return f"Task {task_id} deleted."
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
        self._save(task)
        return json.dumps(task, indent=2)

    def list_all(self) -> str:
        tasks = [json.loads(f.read_text()) for f in sorted(TASKS_DIR.glob("task_*.json"))]
        if not tasks:
            return "No tasks."
        lines = []
        for task in tasks:
            m = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(task["status"], "[?]")
            owner = f" @{task['owner']}" if task.get("owner") else ""
            blocked = f" (blocked by: {task['blockedBy']})" if task.get("blockedBy") else ""
            lines.append(f"{m} #{task['id']}: {task['subject']}{owner}{blocked}")
        return "\n".join(lines)

    def claim(self, task_id: int, owner: str) -> str:
        task = self._load(task_id)
        task["owner"] = owner
        task["status"] = "in_progress"
        self._save(task)
        return f"Claimed task {task_id} for {owner}."


# === SECTION: compression ===
def estimate_tokens(messages: list[MessageParam]) -> int:
    return len(json.dumps(messages, default=str)) // 4

# Layer 1: micro_compact - replace old tool results with placeholders
def micro_compact(messages: list[MessageParam]) -> list[MessageParam]:
    # Collect (msg_index, part_index, tool_result_dict) for all tool_result entries
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))
    if len(tool_results) <= KEEP_RECENT:
        return messages
    # Find tool_name for each result by matching tool_use_id in prior assistant messages
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name
    # Clear old results (keep last KEEP_RECENT)
    to_clear = tool_results[:-KEEP_RECENT]
    for _, _, result in to_clear:
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:
            tool_id = result.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            result["content"] = f"[Previous: used {tool_name}]"
    return messages

# Layer 2: auto_compact - save transcript, summarize, replace messages
def auto_compact(messages: list[MessageParam]) -> list[MessageParam]:
    # Save full transcript to disk
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    # summarize
    conversation_text = json.dumps(messages, default=str)[:80000]
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": """Summarize this conversation for continuity. Include: 
                    1) What was accomplished, 2) Current state, 3) Key decisions made. 
                    Be concise but preserve critical details.\n\n""" + conversation_text}],
        max_tokens=2000,
    )
    summary = response.content[0].text
    # Replace all messages with compressed summary
    return [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. I have the context from the summary. Continuing."},
    ]

# === SECTION: background ===
class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self.notifications = Queue()

    def run(self, command: str, timeout: int = 120) -> str:
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {"status": "running", "command": command, "result": None}
        threading.Thread(target=self._exec, args=(task_id, command, timeout), daemon=True).start()
        return f"Background task {task_id} started: {command[:80]}"

    def _exec(self, task_id: str, command: str, timeout: int):
        try:
            r = subprocess.run(command, shell=True, cwd=WORKDIR, timeout=timeout, capture_output=True, text=True)
            output = (r.stdout + r.stderr).strip()
            self.tasks[task_id].update({"status": "completed", "result": output or "(No output)"})
        except Exception as e:
            self.tasks[task_id].update({"status": "error", "result": f"Error: {str(e)}"})
        self.notifications.put({"task_id": task_id, "status": self.tasks[task_id]["status"], 
                                "result": self.tasks[task_id]["result"][:500]})

    def check(self, task_id: str = None) -> str:
        if task_id:
            task = self.tasks.get(task_id)
            return f"[{task['status']}] {task.get('result', '(running)')}" if task else f"Unknown: {task_id}"
        return "\n".join(f"{k}: [{v['status']}] {v['command'][:60]}" 
                         for k, v in self.tasks.items()) or "No background tasks."

    def drain(self) -> list:
        notifs = []
        while not self.notifications.empty():
            notifs.append(self.notifications.get_nowait())
        return notifs


# === SECTION: messaging ===
class MessageBus:
    def __init__(self) -> None:
        INBOX_DIR.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str, msg_type: str = "message", extra: dict = None) -> str:
        msg = {"type": msg_type, "from": sender, "content": content, "timestamp": time.time()}
        if extra:
            msg.update(extra)
        with open(INBOX_DIR / f"{to}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        path = INBOX_DIR / f"{name}.jsonl"
        if not path.exists():
            return []
        msgs = [json.loads(l) for l in path.read_text().strip().splitlines() if l]
        path.write_text("")
        return msgs

    def broadcast(self, sender: str, content: str, names: list) -> str:
        count = 0
        for n in names:
            if n != sender:
                self.send(sender, n, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


# === SECTION: team ===
class TeammateManager:
    def __init__(self, bus: MessageBus, task_mgr: TaskManager) -> None:
        TEAM_DIR.mkdir(exist_ok=True)
        self.bus = bus
        self.task_mgr = task_mgr
        self.config_path = TEAM_DIR / "config.json"
        self.config = self._load()
        self.threads = {}

    def _load(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find(self, name: str) -> dict:
        for member in self.config["members"]:
            if member["name"] == name:
                return member
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save()
        threading.Thread(target=self._loop, args=(name, role, prompt), daemon=True).start()
        return f"Spawned '{name}' (role: {role})"

    def _set_status(self, name: str, status: str) -> None:
        member = self._find(name)
        if member:
            member["status"] = status
            self._save()

    def _loop(self, name: str, role: str, prompt: str) -> None:
        team_name = self.config["team_name"]
        sys_prompt = f"""You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. 
                         Use idle when done with current work. You may auto-claim tasks."""
        messages = [{"role": "user", "content": prompt}]
        tool_names = {"bash", "read_file", "write_file", "edit_file", "send_message", "idle", "claim_task"}
        tools = [t for t in TOOLS if t["name"] in tool_names]
        while True:
            # -- WORK PHASE --
            for _ in range(50):
                inbox = self.bus.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append({"role": "user", "content": json.dumps(msg)})
                try:
                    response = client.messages.create(model=MODEL, system=sys_prompt, messages=messages, 
                                                      tools=tools, max_tokens=8192)
                except Exception:
                    self._set_status(name, "shutdown")
                    return
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    break
                results = []
                idle_requested = False
                for block in response.content:
                    if block.type == "tool_use":
                        if block.name == "idle":
                            idle_requested = True
                            output = "Entering idle phase."
                        elif block.name == "claim_task":
                            output = self.task_mgr.claim(block.input["task_id"], name)
                        elif block.name == "send_message":
                            output = self.bus.send(name, block.input["to"], block.input["content"])
                        else:
                            dispatch = {k: v for k, v in TOOL_HANDLERS.items() 
                                        if k in {"bash", "read_file", "write_file", "edit_file"}}
                            output = dispatch.get(block.name, lambda **kw: "Unknown")(**block.input)
                        print(f"  [{name}] {block.name}: {str(output)[:120]}")
                        results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                messages.append({"role": "user", "content": results})
                if idle_requested:
                    break
            # -- IDLE PHASE: poll for messages and unclaimed tasks --
            self._set_status(name, "idle")
            resume = False
            for _ in range(IDLE_TIMEOUT // max(POLL_INTERVAL, 1)):
                time.sleep(POLL_INTERVAL)
                inbox = self.bus.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break
                unclaimed = []
                for f in sorted(TASKS_DIR.glob("task_*.json")):
                    t = json.loads(f.read_text())
                    if t.get("status") == "pending" and not t.get("owner") and not t.get("blockedBy"):
                        unclaimed.append(t)
                if unclaimed:
                    task = unclaimed[0]
                    self.task_mgr.claim(task["id"], name)
                    # Identity re-injection for compressed contexts
                    if len(messages) <= 3:
                        messages.insert(0, {"role": "user", "content":
                            f"<identity>You are '{name}', role: {role}, team: {team_name}.</identity>"})
                        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})
                    messages.append({"role": "user", "content": f"""<auto-claimed>
                        Task #{task['id']}: {task['subject']}\n{task.get('description', '')}</auto-claimed>"""
                    })
                    messages.append({"role": "assistant", "content": f"Claimed task #{task['id']}. Working on it."})
                    resume = True
                    break
            if not resume:
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")
    
    def list_all(self) -> str:
        if not self.config["members"]: 
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for member in self.config["members"]:
            lines.append(f"  {member['name']} ({member['role']}): {member['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [member["name"] for member in self.config["members"]]

# === SECTION: shutdown + plan tracking ===
shutdown_requests = {}
plan_requests = {}

# === SECTION: shutdown_protocol ===
def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send("lead", teammate, "Please shut down.", "shutdown_request", {"request_id": req_id})
    return f"Shutdown request {req_id} sent to '{teammate}'"

# === SECTION: plan_approval ===
def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    req = plan_requests.get(request_id)
    if not req: return f"Error: Unknown plan request_id '{request_id}'"
    req["status"] = "approved" if approve else "rejected"
    BUS.send("lead", req["from"], feedback, "plan_approval_response",
             {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"Plan {req['status']} for '{req['from']}'"

# === SECTION: global instances ===
TODO = TodoManager()
SKILLS = SkillLoader(SKILLS_DIR)
TASKS = TaskManager()
BG = BackgroundManager()
BUS = MessageBus()
TEAM = TeammateManager(BUS, TASKS)


# === SECTION: system prompt ===
SYSTEM = f"""
You are a coding agent at {WORKDIR}. Use tools to solve tasks. 
Prefer task_create/task_update/task_list for multi-step work. 
Use todo for short checklists. Mark in_progress before starting, completed when done. 
Use task for subagent delegation to explore unknown topics or subtasks. 
Use load_skill for specialized knowledge before tacking unfamiliar topics. 
Use background_run for long-running commands.
Spawn teammates and communicate via inboxes.
Skills available: {SKILLS.descriptions()}
"""


# === SECTION: tool dispatch===
TOOLS: list[ToolParam] = [
    {"name": "bash", "description": "Run a shell command.", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "todo", "description": "Update task tracking list.", "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "activeForm": {"type": "string"}}, "required": ["id", "content", "status", "activeForm"]}}}, "required": ["items"]}},
    {"name": "task", "description": "Spawn a subagent for isolated exploration or work.", "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}, "agent_type": {"type": "string", "enum": ["Explore", "general-purpose"]}}, "required": ["prompt"]}},
    {"name": "load_skill", "description": "Load specialized knowledge by name.", "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "skill name to load"}}, "required": ["name"]}},
    {"name": "task_create", "description": "Create a new task.", "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update a task's status or dependencies.", "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "add_blocked_by": {"type": "array", "items": {"type": "integer"}}, "add_blocks": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks with status summary.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get full details of a task by ID.", "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    {"name": "compress", "description": "Manually compress conversation context.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "background_run", "description": "Run command in background thread.", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}},
    {"name": "check_background", "description": "Check background task status.", "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}}},
    {"name": "spawn_teammate", "description": "Spawn a persistent autonomous teammate.", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate.", "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send message to all teammates.", "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
    {"name": "shutdown_request", "description": "Request a teammate to shut down.", "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    {"name": "plan_approval", "description": "Approve or reject a teammate's plan.", "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}},
    {"name": "idle", "description": "Enter idle state.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "claim_task", "description": "Claim a task from the board.", "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}}
]


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo": lambda **kw: TODO.update(kw["items"]),
    "task": lambda **kw: run_subagent(kw["prompt"], kw.get("agent_type", "Explore")),
    "load_skill": lambda **kw: SKILLS.load(kw["name"]),
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw["description"]),
    "task_update": lambda **kw: TASKS.update(
        kw["task_id"], kw.get("status"), kw.get("add_blocked_by"), kw.get("add_blocks")
    ),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
    "compress": lambda **kw: "Compressing...",
    "background_run": lambda **kw: BG.run(kw["command"], kw.get("timeout", 120)),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
    "spawn_teammate":   lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":   lambda **kw: TEAM.list_all(),
    "send_message":     lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":       lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":        lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request": lambda **kw: handle_shutdown_request(kw["teammate"]),
    "plan_approval":    lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    "idle":             lambda **kw: "Lead does not idle.",
    "claim_task":       lambda **kw: TASKS.claim(kw["task_id"], "lead"),
}


# === SECTION: print function===
def print_todo(output: object):
    print("\n\033[33m=== 📋 INITIAL PLAN (Todo List) ===\033[0m")
    print(output)
    print("\033[33m=================================\033[0m\n")

def print_tool_call(block: ContentBlock, output: object):
    print("\033[32m--------------------\033[0m")
    print(f"\033[32m> {block.name}: {output}\033[0m")
    print("\033[32m--------------------\033[0m")

def print_thought(block: ContentBlock):
    print(f"\033[90m[Model Thought]: {block.text}\033[0m")

def print_final_response(content: list[ContentBlock]):
    print("\n final response: \n")
    if isinstance(content, list):
        for block in content:
            if hasattr(block, "text"):
                print(f"\033[35m{block.text}\033[0m")
            else:
                print(f"\033[35m{block}\033[0m")


# === SECTION: Agent Loop ===
def agent_loop(messages: list[MessageParam]) -> list[ContentBlock]:
    rounds_since_todo = 0
    while True:
        print(f"\033[36mthinking...\n\033[0m")
        # compress
        micro_compact(messages)
        if estimate_tokens(messages) > TOKEN_THRESHOLD:
            print("[auto-compact triggered]")
            messages[:] = auto_compact(messages)
        # background notifications
        notifs = BG.drain()
        if notifs:
            txt = "\n".join(f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs)
            messages.append({"role": "user", "content": f"<background-results>\n{txt}\n</background-results>"})
            messages.append({"role": "assistant", "content": "Noted background results."})
        # inbox
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({"role": "user", "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>"})
            messages.append({"role": "assistant", "content": "Noted inbox messages."})
        # llm call
        response: Message = client.messages.create(
            model=MODEL, 
            system=SYSTEM, 
            messages=messages, 
            max_tokens=8192, 
            tools=TOOLS
        )

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response.content
        # tools execution
        results: list = []
        used_todo = False
        manual_compress = False
        for block in response.content:
            if block.type == "text":
                print_thought(block)

            if block.type == "tool_use":
                if block.name == "compress":
                    manual_compress = True
                    output = "Compressing..."

                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print_tool_call(block, output)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

                if block.name == "todo":
                    used_todo = True
                    print_todo(output)

        # todo reminder
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if TODO.has_open_items() and rounds_since_todo >= 3:
            results.insert(0, {"type": "text", "text": "<reminder>Update your todos.</reminder>"})
        messages.append({"role": "user", "content": results})
        # manual compress
        if manual_compress:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


# === SECTION: Main ===
if __name__ == "__main__":
    history: list[MessageParam] = []
    while True:
        try:
            query: str = input("\033[36mYou >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/compact":
            if history:
                print("[manual compact via /compact]")
                history[:] = auto_compact(history)
            continue
        if query.strip() == "/compress":
            if history:
                print("[manual compress via /compress]")
                history[:] = auto_compact(history)
            continue
        if query.strip() == "/tasks":
            print(TASKS.list_all())
            continue
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue

        history.append({"role": "user", "content": query})
        resp: list[ContentBlock] = agent_loop(history)
        print_final_response(resp)
        print()
