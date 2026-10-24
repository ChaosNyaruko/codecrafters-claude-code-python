import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import re

from pydantic import BaseModel
from typing import Literal 
from typing import Tuple 
from typing import Dict 
from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

local = os.getenv("CC_LOCAL")
model = "anthropic/claude-haiku-4.5" if not local else "qwen-plus"

class PropertySchema(BaseModel):
    type: str
    description: str

class ParameterSchema(BaseModel):
    type: Literal["object"] = "object"
    props: dict[str, PropertySchema]
    required: list[str]

class FunctionDefinition(BaseModel):
    name: str
    description: str
    params: ParameterSchema

class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition

class WriteTool(FunctionTool):
    def __init__(self):
        super().__init__(
            type = "function",
            function = FunctionDefinition(
                name = "Write",
                description = "Write content to a file",
                params = ParameterSchema(
                        type = "object",
                        props = {
                            "file_path": PropertySchema(type="string", description="The path of the file to write to"),
                            "content":  PropertySchema(type="string", description="The content to write to the file"),
                        },
                        required = [ "file_path", "content" ],
                )
            )
        )
    def __call__(self, args: dict):
        try:
            args = args.get("parameter") or args.get("parameters") or args
            args = list(args.values())
            print("write args: ", args, file=sys.stderr)
            filename = args[0]
            content = args[1]
            with open(filename, 'w') as file:
                file.write(content)
        except Exception as e:
            print("exception: write ", args, e)
            return f"Write error, {e}"
        return f"file {filename} written successfully"

class ReadTool(FunctionTool):
    def __init__(self):
        super().__init__(
            type = "function",
            function = FunctionDefinition(
                name = "Read",
                description = "Read and return the contents of a file",
                params = ParameterSchema(
                        type = "object",
                        props = {"file_path": PropertySchema(type="string", description="The path to the file to read")},
                        required = [ "file_path" ],
                )
            )
        )
    def __call__(self, args):
        args = args.get("parameter") or args.get("parameters") or args
        print("read args: ", args, file=sys.stderr)
        filename = args
        if not filename:
            return "no file name provided"
        try:
            with open(filename, 'r') as file:
                content = file.read()
        except Exception as e:
            print("exception: read ", args, e, file=sys.stderr)
            return ""
        return content

class BashTool(FunctionTool):
    def __init__(self):
        super().__init__(
            type = "function",
            function = FunctionDefinition(
                name = "Bash",
                description = "Execute a shell command",
                params = ParameterSchema(
                        type = "object",
                        props = {"command": PropertySchema(type="string", description="The command to execute")},
                        required = [ "command" ],
                )
            )
        )
    def __call__(self, args):
        args = args.get("parameter") or args.get("parameters") or args.get("command")
        print("bash args: ", args, file=sys.stderr)
        command = args
        if not command:
            return "no command provided"
        try:
            if local:
                return f"executed command: {command}"
            else:
                result = subprocess.run(["bash", "-c", command], capture_output = True)
                if result.returncode != 0:
                    return f"exit code: {result.returncode}, stderr {result.stderr}"
                else:
                    return f"sucess, output\n: {result.stdout}"
        except Exception as e:
            print("exception: bash ", args, e, file=sys.stderr)
            return "error"

tools = [
    ReadTool().model_dump(),
    WriteTool().model_dump(),
    BashTool().model_dump()
]

func_tools_map = {"Read": ReadTool(), "Write": WriteTool(), "Bash": BashTool()}


def build_system_prompt() -> Tuple[str, Dict]:
    # loal skill, and we only load project scoped skills
    skill_path = Path(".claude/skills")
    prompt = f"You have access to the following skill:\n\n"
    skill_files = skill_path.glob("*/SKILL.md")
    skills = {}

    for e in skill_files:
        v = e.read_text()
        # parse front matter
        if not v.startswith("---\n"):
            continue
        _, meta, body = v.split("---\n", 2)

        import yaml
        meta = yaml.safe_load(meta)

        name = meta.get("name")
        desc = meta.get("description")
        s = f"- {name}: {desc}\n"
        skills[name] = {"meta": meta, "body": body}
        prompt += s

    return (prompt, skills)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    system_prompt, skills = build_system_prompt()
    user = []
    # NOTE: we only support invoking ONE skill at one time
    stack_separation = False
    invoked = False
    skill_args = []
    for w in args.p.split():
        print("processing:", w)
        if stack_separation:
            skill_args.append(w)
            continue
        if w.startswith("/"):
            slashcmd = w[1:]
            is_a_skill = slashcmd in skills
            if is_a_skill:
                invoked = True
                user.append(skills[slashcmd].get("body"))
            else:
                # /apple /non-xxx
                if invoked:
                    skill_args.append(w)
                # /non-xxxx /apple
                else:
                    user.append(w)
                stack_separation = True
        else:
            if invoked:
                stack_separation = True
                skill_args.append(w)
            else:
                user.append(w)

    user = " ".join(user)
    # print("user before args:", user)

    all_args = " ".join(skill_args)
    # print("skill args", all_args)

    ALL_ARGS_RE = re.compile(r"\$ARGUMENTS")
    POSITIONAL_ARG_RE = re.compile(r"\$(\d+)")

    user = ALL_ARGS_RE.sub(all_args, user)
    # print("user after trying sub $ARGUMENTS:", user)
    user = POSITIONAL_ARG_RE.sub(lambda m: skill_args[int(m.group(1))], user)

    # print("user after trying sub $0/1/2:", user)

    # exit(1)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools = tools,
        )

        if not response.choices or len(response.choices) == 0:
            raise RuntimeError("no choices in response")

        print(f"response: {response}", file=sys.stderr)
        if response.choices[0].finish_reason == "tool_calls":
            tool_calls = response.choices[0].message.tool_calls
            role = response.choices[0].message.role
            content = response.choices[0].message.content
            messages.append(response.choices[0].message)
            for tool_call in tool_calls:
                tool_call_id = tool_call.id
                if tool_call.type == "function":
                    name, args = tool_call.function.name, tool_call.function.arguments
                    args = json.loads(args)
                    if name not in func_tools_map:
                        raise RuntimeError("func tool {name} not found")
                    tool_result = func_tools_map[name](args)
                    print(f"tool_call result: {tool_call}, result: {tool_result}", file=sys.stderr)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result,
                    })
                else:
                    raise RuntimeError("we don't have non-function tools yet")
        else:
            print(response.choices[0].message.content)
            break


if __name__ == "__main__":
    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)
    main()
