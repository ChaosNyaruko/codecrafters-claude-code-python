[![progress-banner](https://backend.codecrafters.io/progress/claude-code/5ff84de3-42b0-433d-b3e3-1e56dfb1c94e)](https://app.codecrafters.io/users/codecrafters-bot?r=2qF)

This is a starting point for Python solutions to the
["Build Your own Claude Code" Challenge](https://codecrafters.io/challenges/claude-code).

Claude Code is an AI coding assistant that uses Large Language Models (LLMs) to
understand code and perform actions through tool calls. In this challenge,
you'll build your own Claude Code from scratch by implementing an LLM-powered
coding assistant.

Along the way you'll learn about HTTP RESTful APIs, OpenAI-compatible tool
calling, agent loop, and how to integrate multiple tools into an AI assistant.

**Note**: If you're viewing this repo on GitHub, head over to
[codecrafters.io](https://codecrafters.io) to try the challenge.

# Passing the first stage

The entry point for your `claude-code` implementation is in `app/main.py`. Study
and uncomment the relevant code, and submit to pass the first stage:

```sh
codecrafters submit
```

# Stage 2 & beyond

Note: This section is for stages 2 and beyond.

1. Ensure you have `uv` installed locally.
2. Run `./your_program.sh` to run your program, which is implemented in
   `app/main.py`.
3. Run `codecrafters submit` to submit your solution to CodeCrafters. Test
   output will be streamed to your terminal.

# 踩坑记录：LSP - Repair edits with diagnostics (#ql9)

这一关的描述和 tester 的判定逻辑之间有不少没写明的地方，下面是实际踩过的坑。文中的时序片段都来自真实运行时的
`TRACE` 日志（为了排版有删减）。

## 先理清几个概念

- **cursor**：tester 的 LSP 代理（`CODECRAFTERS_LSP_URL`）把 server 推送的所有通知按顺序追加进一个日志里。
  `notifications(after=cursor)` 返回 cursor 之后的新通知和一个新 cursor。新 cursor 下次一定要传回去，
  不管这一批是空的，还是全是 `window/logMessage` 这类无关通知。**"队列"在代理那边**，我们这边只需要记一个 cursor。
- **每条 `publishDiagnostics` 都是该文件的完整诊断**，不是增量（LSP 规范："newly pushed diagnostics always replace
  previously pushed diagnostics"）。所以要整体替换，不能 append。空数组 `[]` 的意思是"这个文件现在没问题了"。
  同一版本可能推送好几次，所以还要按文件去重，同样的问题不要每轮都发给模型。
- **何时发给模型**：每次 `chat.completions.create()` 之前读一次队列，把新增的诊断追加到 `messages` 里最后一条
  tool 消息上。Write 自己的返回内容不变。

## 坑 1：先检查版本，再解析内容

tester 会故意发一条格式不完整的旧报告（诊断里没有 `range`）：

```
publishDiagnostics pricing.py report_v=1 our_v=2 STALE->drop [(None, 'Stale diagnostic must be ignored')]
```

如果先用 pydantic 校验整个 params、再比较版本，程序会直接因为 ValidationError 退出。正确的顺序是：
method 过滤 → 取 `uri` / `version`（`version` 可能没有，用 `.get()`，**不要补默认值**）→ 版本检查
→ 通过后才解析 `diagnostics`。

## 坑 2：tester 眼里的 "current" 是指报告**发出时**是当前版本

题目表格写的是 "Version matches the last version sent for that file"，但没说是**什么时候**的 last version。
按字面理解，很自然会在读取时去比较，而 tester 要求的是：报告发出的那一刻版本匹配，就必须送到模型那里。
另外，在两次请求之间被同一文件的后续报告覆盖掉的诊断，也要送到。

模型经常**一次回复里连续 Write 好几个文件**，这正是出问题的地方。

### 失败方式 A：报告带了 version，读到时文件已经被改过了（被误当成旧报告丢掉）

```
MODEL RETURNED 3 tool call(s) in one response
TOOL CALL Write pricing.py   → didChange pricing.py v2
TOOL CALL Write checkout.py  → didChange checkout.py v2      ← checkout 的版本号已经变了
TOOL CALL Write reports.py   → didOpen reports.py v1
READ [before-request] cursor 3 -> 10, 7 notification(s)
   publishDiagnostics checkout.py report_v=1 our_v=2 STALE->drop [5:12 'Type Price is not assignable to return type int']
   publishDiagnostics checkout.py report_v=2 our_v=2 accept []
APPEND to last tool result: (nothing)                         ← 模型永远看不到这个错误
```

server 收到 pricing 的改动后，基于 checkout **v1** 算出了报错。这份报告在发出时是当前的，
但我们读到它时 checkout 已经是 v2 了，于是被丢掉。

### 失败方式 B：报告没带 version，被接受了，又被同一批里的下一份报告覆盖

```
READ [before-request] cursor 10 -> 24, 14 notification(s)
   publishDiagnostics checkout.py report_v=None our_v=2 accept [5:12 'Type Price is not assignable ...']
   publishDiagnostics checkout.py report_v=1    our_v=2 STALE->drop []
   publishDiagnostics checkout.py report_v=None our_v=2 accept []    ← 覆盖了上面的 [A]
```

如果"算新增"是在请求前、用"每个文件的最新报告"去减"已展示集合"来做，那么同一批里后来的 `[]`
会把 `[A]` 覆盖掉，模型同样看不到。

### 修法：两处改动缺一不可

1. **改版本号之前先读一次队列**：`did_change` 里，在 `version += 1` 之前调用一次读取，让基于旧版本的报告按照
   "当时的版本"被判定。
2. **在接受报告的那一刻就算出新增，放进 `pending`**：
   - `shown[file]`：这个文件最近一次被接受的报告，作为去重基准，整体替换。
   - `pending[file]`：被接受的报告里新出现、但还没发给模型的诊断。请求前格式化发出，然后清空。

修好之后的时序：

```
TOOL CALL Write checkout.py
READ [before-didChange] cursor 10 -> 13, 3 notification(s)
   publishDiagnostics pricing.py  report_v=1    our_v=2 STALE->drop [(None, 'Stale diagnostic must be ignored')]
   publishDiagnostics checkout.py report_v=None our_v=1 accept [5:12 'Type Price ...']   ← 进入 pending
didChange checkout.py v2
READ [before-request] cursor 13 -> 16, 3 notification(s)
   publishDiagnostics checkout.py report_v=1 our_v=2 STALE->drop []
APPEND to last tool result: '<new-diagnostics>... checkout.py:\n  [Line 5:12] Type Price is not assignable to return type int</new-diagnostics>'
```

对照实验（只改诊断逻辑，其他都一样）：没有 pending 和提前读取的版本，7 次里过了 2 次，失败全是
"the model never saw the server's diagnostic for checkout.py at 5:12"；有的版本，6 次全部通过。
结果"时过时不过"，取决于模型这次是否在同一个回复里连续写了多个文件。

> 关于合理性：按 tester 的做法，模型可能会收到已经过时的诊断（它刚把那一行修好）。这是"宁可多报，不能漏报"的取舍。
> 在真实的 agent 里，可以在文字上注明"该报告针对的是更早的版本，可能已经修复，请先 Read 确认"。

## 坑 3：和诊断无关、但同样会导致"时过时不过"的 bug

- **Bash tool 丢掉了 stderr**：`unittest` 的报告（`Ran 5 tests` / `OK`）是写到 **stderr** 的。原来的实现在退出码为 0 时只返回
  stdout，模型拿到的是空输出，于是编造测试结果（tester 报 "run the project tests through Bash ..."），或者反复重跑直到超时。
  修法：无论退出码是多少，都返回退出码、stdout 和 stderr，用 `text=True` 解码。
- **改完文件后行号变了**：第二个场景的 pricing.py 在 `def total` 上面多了一行注释，模型改完后照搬了旧的行号
  （第 8 行，实际在第 9 行）去做 findReferences，tester 报 "missing findReferences ... at the current definition"。
  修法：LSP tool 的结果里回显被查询位置当前那一行的源码，并用 `^` 标出列；tool 描述里也强调编辑后要重新确定位置。

## 调试建议

- 在怀疑模型之前，先看它**实际收到的 tool result** 是什么。
- 给关键事件单独加 trace（读取时 cursor 从多少到多少、每条报告的 `report_v` / `our_v` 和接受还是丢弃、didOpen/didChange 的版本号、
  每次请求追加了什么），输出到 stderr。tester 会把 stderr 一起回显出来，比打印整个 response 对象好读得多。

# References
[LSP intro](https://code.visualstudio.com/api/language-extensions/language-server-extension-guide)
![What is LSP](https://code.visualstudio.com/assets/api/language-extensions/language-server-extension-guide/lsp-languages-editors.png)
