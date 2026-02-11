# Codex Python SDK

Embed the Codex agent in Python workflows and apps.

The Python SDK uses `codex app-server` and speaks JSON-RPC over stdio. It does not use `codex exec`.

## Installation (local development)

```bash
cd sdk/python
python -m pip install -e .[dev]
```

## Quickstart

```python
from openai_codex_sdk import Codex

codex = Codex()
thread = codex.start_thread()
turn = await thread.run("Diagnose the failing test and propose a fix.")

print(turn.final_response)
print(turn.items)
```

Call `run()` repeatedly on the same `Thread` instance to continue that conversation.

```python
next_turn = await thread.run("Implement the fix.")
```

## Streaming responses

`run()` buffers events until the turn finishes. To react to intermediate progress, use
`run_streamed()`, which returns an async iterator of structured events.

```python
from openai_codex_sdk import Codex, ItemCompletedEvent, TurnCompletedEvent

codex = Codex()
thread = codex.start_thread()
streamed = await thread.run_streamed("Diagnose the failure and propose a fix.")

async for event in streamed.events:
    if isinstance(event, ItemCompletedEvent):
        print("item:", event.item)
    elif isinstance(event, TurnCompletedEvent):
        print("usage:", event.usage)
```

## Structured output

Pass `output_schema` in `TurnOptions`; the SDK forwards it to `turn/start.outputSchema`.

```python
from openai_codex_sdk import Codex, TurnOptions

schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "status": {"type": "string", "enum": ["ok", "action_required"]},
    },
    "required": ["summary", "status"],
    "additionalProperties": False,
}

codex = Codex()
thread = codex.start_thread()
turn = await thread.run("Summarize repository status", TurnOptions(output_schema=schema))
print(turn.final_response)
```

## Attaching images

Provide structured input entries when you need to include images alongside text.

```python
turn = await thread.run(
    [
        {"type": "text", "text": "Describe these screenshots"},
        {"type": "local_image", "path": "./ui.png"},
        {"type": "local_image", "path": "./diagram.jpg"},
    ]
)
```

## Resuming an existing thread

Threads are persisted by Codex in `~/.codex/sessions`.

```python
thread = codex.resume_thread("your-thread-id")
await thread.run("Continue from here.")
```

## Working directory controls

Use thread options to set model, sandbox mode, and working directory defaults.

```python
thread = codex.start_thread(
    {
        "model": "gpt-5-codex",
        "sandbox_mode": "workspace-write",
        "working_directory": "/path/to/project",
    }
)
```

`skip_git_repo_check` is an exec-transport flag and is not available via `app-server`; this
SDK raises an error if you set it.

## Environment and config overrides

By default, the SDK inherits `os.environ`. Set `env` in `CodexOptions` to provide a full
environment override.

```python
codex = Codex(
    {
        "env": {"PATH": "/usr/local/bin"},
        "config": {
            "show_raw_agent_reasoning": True,
            "sandbox_workspace_write": {"network_access": True},
        },
    }
)
```

## Approval behavior

When app-server asks for command/file-change approvals, this prototype SDK auto-accepts those
approval requests to keep non-interactive runs moving.
