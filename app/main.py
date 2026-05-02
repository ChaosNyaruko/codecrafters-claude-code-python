import argparse
import os
import sys

from pydantic import BaseModel
from typing import Literal 
from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

class PropertySchema(BaseModel):
    type: str
    desc: str

class ParameterSchema(BaseModel):
    type: Literal["object"] = "object"
    props: dict[str, PropertySchema]
    required: list[str]

class FunctionDefinition(BaseModel):
    name: str
    desc: str
    props: ParameterSchema

class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition

class ReadTool(FunctionTool):
    def __init__(self):
        super().__init__(
            type = "function",
            function = FunctionDefinition(
                name = "Read",
                desc = "Read and return the contents of a file",
                props = ParameterSchema(
                        type = "object",
                        props = {"file_path": PropertySchema(type="string", desc="The path to the file to read")},
                        required = [ "file_path" ],
                )
            )
        )

tools = [
    ReadTool().model_dump()
]

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

    print(chat.choices[0].message.content)


if __name__ == "__main__":
    main()
