import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import re

from pydantic import BaseModel

from typing import cast
from typing import Iterable
from typing import Literal
from typing import Tuple
from typing import Dict
from typing import MutableSequence
from openai import OpenAI
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

local = os.getenv("CC_LOCAL")
model = "anthropic/claude-haiku-4.5" if not local else "qwen-plus"

class Agent:
    def __init__(self, skills: Dict[str, Skill], tools: Iterable[ChatCompletionToolUnionParam], client: OpenAI, system_prompt):
        self.skills = skills
        self.tools = tools
        self.client = client
        self.system_prompt = system_prompt

    def process_user_input(self, prompt: str) -> Tuple[MutableSequence, bool]:
        # we iterate the user input twice, the first round is to determin whether the skill should be expanded:
        # See "Notes" in https://app.codecrafters.io/courses/claude-code/stages/mj2
        #   "A skill that asks for a subagent ends a stacking run, so it's never expanded alongside another."
        sub_agent_called = False
        sub_body = []
        for w in prompt.split():
            if w.startswith("/"):
                slashcmd = w[1:]
                is_a_skill = slashcmd in self.skills
                if is_a_skill and self.skills[slashcmd].fork:
                    sub_agent_called = True
                    sub_body.append(self.skills[slashcmd].body)
                    # NOTE: we only support ONE subagent for skill now
                    break
        if sub_agent_called:
            return sub_body, True

        user = []
        # NOTE: we only support invoking ONE skill at one time
        stack_separation = False
        invoked = False
        skill_args = []
        for w in prompt.split():
            # print("processing:", w)
            if stack_separation:
                skill_args.append(w)
                continue
            if w.startswith("/"):
                slashcmd = w[1:]
                is_a_skill = slashcmd in self.skills
                if is_a_skill:
                    invoked = True
                    user.append(self.skills[slashcmd].body)
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
        messages = list()

        for name, skill in self.skills.items():
            msg = f"Skill: {name} (located at {skill.path})"
            messages.append(ChatCompletionUserMessageParam({"role": "user", "content": msg}))

        messages.append(ChatCompletionUserMessageParam({"role": "user", "content": user}))
        return messages, False

    def run_sync_loop(self, prompt: str) -> str:
        print(f"agent prompt: {prompt}", file=sys.stderr)
        user_messages, delegated = self.process_user_input(prompt)
        messages = list()
        if self.system_prompt:
            messages.append(ChatCompletionSystemMessageParam({"content": self.system_prompt, "role": "system"}))
        if delegated:
            subagent = Agent({}, [], self.client, "")
            # a clean context
            answer = subagent.run_sync_loop("".join(user_messages))
            print(answer)
        else:
            for u in user_messages:
                messages.append(u)
        while True:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools = self.tools,
            )
            if not response.choices or len(response.choices) == 0:
                raise RuntimeError("no choices in response")

            print(f"response: {response}", file=sys.stderr)
            if response.choices[0].finish_reason == "tool_calls":
                tool_calls = response.choices[0].message.tool_calls
                assert tool_calls is not None, "tool call should not be None"
                messages.append(response.choices[0].message)
                for tool_call in tool_calls:
                    tool_call_id = tool_call.id
                    if tool_call.type == "function":
                        name, args = tool_call.function.name, tool_call.function.arguments
                        args = json.loads(args)
                        if name not in func_tools_map:
                            raise RuntimeError("func tool {name} not found")
                        tool_result, stop = func_tools_map[name](args)
                        print(f"tool_call result: {tool_call}, result: {tool_result}", file=sys.stderr)
                        if stop:
                            return tool_result
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tool_result,
                        })
                    else:
                        raise RuntimeError("we don't have non-function tools yet")
            else:
                res = response.choices[0].message.content
                assert res is not None, "agent result should not be None"
                return res

class Skill():
    def __init__(self, name: str, path: Path, meta: str, body: str, fork: bool):
        self.name = name
        self.path = path
        self.meta = meta
        self.body = body
        self.fork = fork

def build_system_prompt() -> Tuple[str, Dict]:
    # loal skill, and we only load project scoped skills
    skill_path = Path(".claude/skills")
    prompt = f"You have access to the following skill:\n\n"
    skill_files = skill_path.glob("*/SKILL.md")
    skills = {}

    for e in skill_files:
        path = e.absolute().relative_to(e.cwd(), walk_up=True).parent
        v = e.read_text()
        # parse front matter
        if not v.startswith("---\n"):
            continue
        _, meta, body = v.split("---\n", 2)

        import yaml
        meta = yaml.safe_load(meta)

        name = meta.get("name")
        desc = meta.get("description")
        fork = meta.get("context", False)
        s = f"- {name}: {desc}\n"
        skills[name] = Skill(name, path, meta, body, fork)
        prompt += s

    # prompt += "\n\nIf a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns.\n"

    return (prompt, skills)

class PropertySchema(BaseModel):
    type: str
    description: str

class ParameterSchema(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, PropertySchema]
    required: list[str]

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: ParameterSchema

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
                parameters = ParameterSchema(
                        type = "object",
                        properties = {
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
            args_list = list(args.values())
            print("write args_list: ", args_list, file=sys.stderr)
            filename = args_list[0]
            content = args_list[1]
            with open(filename, 'w') as file:
                file.write(content)
        except Exception as e:
            print("exception: write ", args, e)
            return f"Write error, {e}", False
        return f"file {filename} written successfully", False

class ReadTool(FunctionTool):
    def __init__(self):
        super().__init__(
            type = "function",
            function = FunctionDefinition(
                name = "Read",
                description = "Read and return the contents of a file",
                parameters = ParameterSchema(
                        type = "object",
                        properties = {"file_path": PropertySchema(type="string", description="The path to the file to read")},
                        required = [ "file_path" ],
                )
            )
        )
    def __call__(self, args):
        print("read args: type: ", args, type(args), file=sys.stderr)
        filename = ""
        for k, v in args.items():
            # a "fuzzy" match for models that are not stable (e.g. haiku)
            if re.compile("param").search(k) or \
                re.compile("name").search(k) or \
                re.compile("file").search(k) or \
                re.compile("path").search(k):
                filename = v
                break

        if not filename:
            return "no file name provided", False
        try:
            with open(Path(filename), 'r') as file:
                content = file.read()
        except Exception as e:
            print("exception: read ", filename, e, file=sys.stderr)
            return "", False
        return content, False

global_system_prompt, global_skills = build_system_prompt()
class SkillTool(FunctionTool):
    def __init__(self):
        super().__init__(
            type = "function",
            function = FunctionDefinition(
                name = "Skill",
                description = "Load a skill's instructions into the conversation",
                parameters = ParameterSchema(
                        type = "object",
                        properties = {
                            "name": PropertySchema(type="string", description="The name of the skill to use"),
                            "args": PropertySchema(type="string", description="Optional arguments for the skill"),
                        },
                        required = [ "name" ],
                )
            )
        )

    def __call__(self, args):
        name = args.get("skill") or args.get("name")
        if not name:
            return "skill name not provided", False
        try:
            if local:
                return f"that's skill result", True
            else:
                sk = global_skills.get(name, "cannot get body of this skill")
                body = sk.body
                if not sk.fork:
                    return f"{body}\n", False
                else:
                    subagent = Agent({}, [], global_client, "")
                    # a clean context
                    answer = subagent.run_sync_loop(body)
                    return answer, True

        except Exception as e:
            print("exception: calling skill", args, e, file=sys.stderr)
            return "error", True

class BashTool(FunctionTool):
    def __init__(self):
        super().__init__(
            type = "function",
            function = FunctionDefinition(
                name = "Bash",
                description = "Execute a shell command",
                parameters = ParameterSchema(
                        type = "object",
                        properties = {"command": PropertySchema(type="string", description="The command to execute")},
                        required = [ "command" ],
                )
            )
        )

    def __call__(self, args):
        args = args.get("parameter") or args.get("parameters") or args.get("command")
        print("bash args: ", args, file=sys.stderr)
        command = args
        if not command:
            return "no command provided", False
        try:
            if local:
                return f"executed command: {command}", False
            else:
                result = subprocess.run(["bash", "-c", command], capture_output = True)
                if result.returncode != 0:
                    return f"exit code: {result.returncode}, stderr {result.stderr}", False
                else:
                    return f"sucess, output\n: {result.stdout}", False
        except Exception as e:
            print("exception: bash ", args, e, file=sys.stderr)
            return "error", False

global_tools : list[ChatCompletionToolUnionParam] = [
    cast(ChatCompletionToolUnionParam, t.model_dump()) for t in (
        ReadTool(),
        WriteTool(),
        BashTool(),
        SkillTool(),
    )
]

global_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

func_tools_map = {"Read": ReadTool(), "Write": WriteTool(), "Bash": BashTool(), "Skill": SkillTool()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    main_agent = Agent(global_skills, global_tools, _client, global_system_prompt)
    print(main_agent.run_sync_loop(args.p))


if __name__ == "__main__":
    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)
    main()
