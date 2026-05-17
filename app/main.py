import argparse
import json
import os
import subprocess
import sys

from pydantic import BaseModel
from typing import Literal 
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
                    return f"exit code: {result.returncode}, stder {result.stderr}"
                else:
                    return result.stdout
        except Exception as e:
            print("exception: bash ", args, e, file=sys.stderr)
            return "error"

tools = [
    ReadTool().model_dump(),
    WriteTool().model_dump(),
    BashTool().model_dump()
]

func_tools_map = {"Read": ReadTool(), "Write": WriteTool(), "Bash": BashTool()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools = tools
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
