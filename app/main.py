import argparse
import os
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

class Property():
    def __init__(self, name, type, desc):
        self.name = name
        self.type = type
        self.desc = desc

class Properties():
    def __init__(self, *args):
        self.properties = dict() # TODO: TypedDict
        for p in args:
            self._add_property(p)
    def _add_property(self, p):
        if p.name in self.properties:
            raise RuntimeError(f"cannot have duplicate props: {p.name}")
        self.properties[p.name] = p


class Parameters():
    def __init__(self, properties, required):
        self.type = "object"
        self.properties = properties
        self.required = required

class FunctionTool:
    def __init__(self, name, description):
        self.type = "function"
        self.name = name
        self.description = description

class ReadTool(FunctionTool):
    def __init__(self):
        super().__init__("Read", "Read and return the contents of a file")
        props = Properties(Property("file_path", "string", "The path to the file to read"))
        self.parameters = Parameters(props, required=["file_path"])

tools = [
     {
      "type": "function",
      "function": {
        "name": "Read",
        "description": "Read and return the contents of a file",
        "parameters": {
          "type": "object",
          "properties": {
            "file_path": {
              "type": "string",
              "description": "The path to the file to read"
            }
          },
          "required": ["file_path"]
        }
      }
    }
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
