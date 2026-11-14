import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import re
from pydantic import BaseModel, RootModel
from traceback import print_exc
from typing import cast
from typing import Iterable
from typing import Literal
from typing import Tuple
from typing import Dict
from typing import MutableSequence
from typing import Optional


from openai import OpenAI
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam

from app.lsp_client import LSPClient

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

local = os.getenv("CC_LOCAL")
debug = os.getenv("DEBUG")
model = "anthropic/claude-haiku-4.5" if not local else "qwen-plus"

def eprint(*args):
    return
    print(*args, file=sys.stderr)

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
        # eprint("user before args:", user)

        all_args = " ".join(skill_args)
        # eprint("skill args", all_args)

        ALL_ARGS_RE = re.compile(r"\$ARGUMENTS")
        POSITIONAL_ARG_RE = re.compile(r"\$(\d+)")

        user = ALL_ARGS_RE.sub(all_args, user)
        # eprint("user after trying sub $ARGUMENTS:", user)
        user = POSITIONAL_ARG_RE.sub(lambda m: skill_args[int(m.group(1))], user)

        # eprint("user after trying sub $0/1/2:", user)

        # exit(1)
        messages = list()

        for name, skill in self.skills.items():
            msg = f"Skill: {name} (located at {skill.path})"
            messages.append(ChatCompletionUserMessageParam({"role": "user", "content": msg}))

        messages.append(ChatCompletionUserMessageParam({"role": "user", "content": user}))
        return messages, False

    def run_sync_loop(self, prompt: str) -> str:
        eprint(f"agent prompt: {prompt}")
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
            global_lsp_client.get_diagnostics_notification()
            if len(messages) > 0 and messages[-1]["role"] == "tool":
                info = global_lsp_client.send_diagnostics()
                messages[-1]["content"] = messages[-1]["content"] + info
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools = self.tools,
            )
            if not response.choices or len(response.choices) == 0:
                raise RuntimeError("no choices in response")

            eprint(f"response: {response}")
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

                        eprint(f"tool_call result: {tool_call}, result: {tool_result}")
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

    # for LSP tools 
    enum: Optional[list[str]] = None
    minimum: Optional[int] = None

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

class LspLinePos(BaseModel):
    line: int
    character: int

class LspRange(BaseModel):
    start: LspLinePos
    end: LspLinePos

    
class LspLocation(BaseModel):
    uri: str
    range: LspRange

LspDefinitionResponse = LspLocation

class LspReferencesResponse(RootModel[list[LspLocation]]):
    def __iter__(self): # pyright: ignore[reportIncompatibleMethodOverride]
        return iter(self.root)
    def __len__(self):
        return len(self.root)
    def __getitem__(self, i):
        return self.root[i]

class FileState():
    def __init__(self):
        self.version = 1

    def inc(self) -> int:
        self.version += 1
        return self.version

class LspDiagnostic(BaseModel):
    range: LspRange
    severity: int | None = None
    message: str

class LspDiagnostics(RootModel[list[LspDiagnostic]]):
    def __iter__(self): # pyright: ignore[reportIncompatibleMethodOverride]
        return iter(self.root)
    def __len__(self):
        return len(self.root)
    def __getitem__(self, i):
        return self.root[i]

class LspNotificationParams(BaseModel):
    uri: str
    version: int | None = None
    diagnostics: LspDiagnostics 

class LspNotification(BaseModel):
    method: str
    params: dict

class LspNotifications(BaseModel):
    cursor: int
    notifications: list[LspNotification]

class LspClient():
    def __init__(self):
        self.client = LSPClient.from_environment()
        self.notified = dict()
        self.next_cursor = 0
        self.shown: dict[str, set] = defaultdict(set)    # file -> last accepted report
        self.pending: dict[str, list] = defaultdict(list)  # file -> new issues not yet sent

    def get_diagnostics_notification(self, timeout_ms: int = 100):
        """Read queued notifications and stage new diagnostics in self.pending.

        Version is checked against the version we sent *at read time*, so this
        must also run right before a file's version is bumped (see did_change).
        """
        if self.client is None:
            return
        resp = self.client.notifications(self.next_cursor, timeout_ms)
        eprint("diag:", resp)
        ns = LspNotifications.model_validate(resp)
        self.next_cursor = ns.cursor  # advance even if nothing below is useful
        for n in ns.notifications:
            if n.method != "textDocument/publishDiagnostics":
                continue
            # Check the version BEFORE validating the diagnostics: stale reports
            # may be malformed and must be ignored without being parsed.
            file = str(Path.from_uri(n.params["uri"]).absolute())
            v = n.params.get("version")
            fs = self.notified.get(file)
            if fs is not None and v is not None and fs.version != v:
                eprint(f"stale report for {file}: sent v{fs.version}, report v{v}")
                continue
            params = LspNotificationParams.model_validate(n.params)
            keys = [(d.range.start.line, d.range.start.character, d.message)
                    for d in params.diagnostics]
            # Only issues absent from the previous report count as new; the
            # latest report replaces the previous one (empty list clears it).
            prev = self.shown[file]
            self.pending[file].extend(k for k in keys if k not in prev)
            self.shown[file] = set(keys)

    def send_diagnostics(self) -> str:
        """Format and drain staged diagnostics; "" if there is nothing new."""
        if not any(self.pending.values()):
            self.pending.clear()
            return ""
        groups = []
        for file, keys in self.pending.items():
            if not keys:
                continue
            lines = [f"  [Line {line + 1}:{char + 1}] {msg}" for line, char, msg in keys]
            groups.append(os.path.relpath(file) + ":\n" + "\n".join(lines))
        self.pending.clear()
        res = ("\n\n<new-diagnostics>The following new diagnostic issues were detected "
               "(use them to repair your edits):\n\n"
               + "\n\n".join(groups) + "</new-diagnostics>")
        eprint("[for model:]", res)
        return res


    def did_change(self, path: Path, text: str):
        assert self.client is not None
        # {
  # "method": "textDocument/didChange",
  # "params": {
    # "textDocument": { "uri": "file:///project/checkout.py", "version": 2 },
    # "contentChanges": [{ "text": "from discounts import total as invoice_total\n\n\ndef checkout_total() -> int:\n    return invoice_total(250, 2)\n" }]
  # }
# }
        p = str(path.absolute())
        fs = self.notified.get(p)
        assert fs is not None
        # Consume reports for the current version before we bump it, otherwise
        # they would look stale by the time we read them.
        self.get_diagnostics_notification()
        ver = fs.inc()
        params = {
            "textDocument": {
                "uri": path.as_uri(), 
                "version": ver,
            },
            # The server uses full synchronization: contentChanges contains one object with the complete text that was saved, including line breaks. No diff or edit range is needed.
            "contentChanges": [
                {
                    "text": text
                }
            ]
        }
        self.client.notify("textDocument/didChange", params)

    def did_open_if_needed(self, path: Path) -> bool:
        assert self.client is not None
        langId = self.client.extension_to_language.get(''.join(path.suffixes))
        if not self.notified.get(str(path)):
            with open(path, 'r') as file:
                text = file.read()
            nt_params = {
                "textDocument": {
                    "uri": path.as_uri(), 
                    "languageId": langId,
                    "version": 1,
                    "text": text,
                },
            }
            self.client.notify("textDocument/didOpen", nt_params)
            self.notified[str(path)] = FileState()
            return True
        return False

global_lsp_client = LspClient()

DEFINITION = "goToDefinition"
REFERENCES = "findReferences"
class LspTool(FunctionTool):
    def __init__(self):
        super().__init__(
            type = "function",
            function = FunctionDefinition(
                name = "LSP",
                description = "Returns the file, line, and character where a symbol is defined.",
                parameters = ParameterSchema(
                        type = "object",
                        properties = {
                            "operation": PropertySchema(
                                type="string", 
                                enum=[DEFINITION, REFERENCES], 
                                description=""
                            ),
                            "filePath":  PropertySchema(
                                type="string", 
                                description="Absolute path, or a path relative to the agent's current working directory."
                            ),
                            "line": PropertySchema(
                                type="integer", 
                                minimum=1, 
                                description="1-based line number of the symbol."
                            ),
                            "character": PropertySchema(
                                type="integer", 
                                minimum=1, 
                                description="1-based character offset in UTF-16 code units (same as character count for ASCII)"),
                        },
                        required = [ "operation", "filePath", "line", "character" ],
                )
            )
        )
    def __call__(self, args: dict):
        try:
            oper = args.get("operation")
            if not oper:
                return "operation not found in lsp tool call", False
            if oper == DEFINITION:
                path = Path(args.get("filePath", "")).absolute()
                line = args.get("line", 1)
                character = args.get("character", 1)
                def_params = {
                    "textDocument": {
                        "uri": path.as_uri(),
                    },
                    "position": {
                        "line": line - 1,
                        "character": character - 1,
                    },
                }
                global_lsp_client.did_open_if_needed(path)
                assert global_lsp_client.client is not None, "lsp client should not be None"

                resp = global_lsp_client.client.request("textDocument/definition", def_params)
                resp = LspDefinitionResponse.model_validate(resp)
                p = Path.from_uri(resp.uri)
                eprint(p)
                return f"Defined in {p}:{resp.range.start.line + 1}:{resp.range.start.character + 1}", False
            elif oper == REFERENCES:
                path = Path(args.get("filePath", "")).absolute()
                line = args.get("line", 1)
                character = args.get("character", 1)
                params = {
                    "textDocument": {
                        "uri": path.as_uri(),
                    },
                    "position": {
                        "line": line - 1,
                        "character": character - 1,
                    },
                    "context": {
                        "includeDeclaration": True,
                    }
                }
                global_lsp_client.did_open_if_needed(path)
                assert global_lsp_client.client is not None, "lsp client should not be None"
                resp = global_lsp_client.client.request("textDocument/references", params)
                resp = LspReferencesResponse.model_validate(resp)
                res = ""
                locs = defaultdict(list[LspLinePos])
                for l in resp:
                    p = str(Path.from_uri(l.uri))
                    line = l.range.start.line + 1
                    col = l.range.start.character + 1
                    locs[p].append(LspLinePos(line=line, character=col))
                if len(locs) == 0:
                    return "No references found", False
                for (file, ls) in locs.items():
                    f = file + "\n"
                    for l in ls:
                        f += f" Line:{l.line}:{l.character}\n"
                    res += f + "\n"
                return res, False
            else:    
                return f"{oper} not implemented yet", False
        except Exception as e:
            print_exc()
            return f"lsp tool error {e}", False

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
            eprint("write args_list: ", args_list)
            filename = args_list[0]
            content = args_list[1]
            with open(filename, 'w') as file:
                file.write(content)
            if global_lsp_client.client is not None:
                p = Path(filename).absolute()
                init = global_lsp_client.did_open_if_needed(p)
                if not init:
                    global_lsp_client.did_change(p, content)
        except Exception as e:
            eprint("exception: write ", args, e)
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
        # eprint("read args: type: ", args, type(args), file=sys.stderr)
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
            eprint("exception: read ", filename, e)
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
            eprint("exception: calling skill", args, e)
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
        eprint("bash args: ", args)
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
            eprint("exception: bash ", args, e)
            return "error", False


global_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

func_tools_map = {"Read": ReadTool(), "Write": WriteTool(), "Bash": BashTool(), "Skill": SkillTool(), "LSP": LspTool()}
global_tools : list[ChatCompletionToolUnionParam] = [
    cast(ChatCompletionToolUnionParam, t.model_dump(exclude_none=True)) for t in func_tools_map.values()
]

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
    eprint("Logs from your program will appear here!")
    main()
