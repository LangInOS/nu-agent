
import os
import subprocess

from typing import Any
from anthropic import Anthropic
from anthropic.types import Message, MessageParam, ToolParam, ContentBlock
from typing import List

from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client: Anthropic = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

MODEL: str | None = os.getenv("MODEL_ID")

if not MODEL:
    raise ValueError("MODEL_ID environment variable not set")

SYSTEM: str = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS : list[ToolParam] = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": { "command": { "type": "string" } },
            "required": ["command"]

        }
    }
]


def run_bash(command: str):
    forbidden = ["rm -fr /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in forbidden):
        return "Error: Dangerous command blocked"

    try:
        result = subprocess.run(command,
                                shell=True,
                                cwd=os.getcwd(),
                                text=True,
                                timeout=120,
                                capture_output=True)

        out = result.stdout + result.stderr

        return out if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- agent loop: calls tools until the llm stops --
def agent_loop(messages: list[MessageParam]) -> List[ContentBlock]:
    while(True):
        print(f"\033[36m thinking...\n\033[0m")
        response: Message = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            max_tokens=8192,
            tools = TOOLS
        )

        messages.append({ "role": "assistant", "content": response.content })

        if response.stop_reason != "tool_use":
            return response.content

        results: list[MessageParam] = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.input['command']}\033[0m")
                print("\033[32m--------------------\033[0m")
                output = run_bash(block.input["command"])
                print(f"\033[32m{output}\033[0m")
                print("\033[32m--------------------\033[0m")
                results.append({ "type": "tool_result", "tool_use_id": block.id, "content": output })

        messages.append({ "role": "user", "content": results })



if __name__ == "__main__":
    history: list[MessageParam] = []
    while(True):
        try:
            query: str = input("\033[36mYou >> \033[0m ")
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



