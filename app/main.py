import argparse
import json
import os
import sys

from pydantic import BaseModel
from typing import Literal 
from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

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
    def __call__(self, args: dict):
        props = list(self.function.params.props.keys())
        akeys = list(args.keys())
        assert akeys[0] == props[0], "the argument is called different"
        with open(list(args.values())[0], 'r') as file:
            content = file.read()
        return content

tools = [
    ReadTool().model_dump()
]

func_tools_map = {"Read": ReadTool()}

model = "anthropic/claude-haiku-4.5" if not os.getenv("CC_LOCAL") else "qwen-plus"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": args.p}],
        tools = tools
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)


    if chat.choices[0].finish_reason == "tool_calls":
        tool_calls = chat.choices[0].message.tool_calls
        print(tool_calls)
        for tool_call in tool_calls:
            if tool_call.type == "function":
                name, args = tool_call.function.name, tool_call.function.arguments
                args = json.loads(args)
                print(name, args)
                if name not in func_tools_map:
                    raise RuntimeError("func tool {name} not found")
                print(func_tools_map[name](args))
            else:
                raise RuntimeError("we don't have non-function tools yet")
    else:
        print(chat.choices[0].message.content)


if __name__ == "__main__":
    main()
