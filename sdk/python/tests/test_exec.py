from __future__ import annotations

import asyncio
import json

import pytest

from openai_codex_sdk.exec import CodexExec, CodexExecArgs, serialize_config_overrides


class FakeStdin:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, value: bytes) -> None:
        self.data.extend(value)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class FakeProcess:
    def __init__(
        self,
        stdout_lines: list[str],
        stderr_text: str = "",
        returncode: int = 0,
    ) -> None:
        self.stdin = FakeStdin()
        self.stdout = asyncio.StreamReader()
        for line in stdout_lines:
            self.stdout.feed_data(line.encode("utf-8") + b"\n")
        self.stdout.feed_eof()

        self.stderr = asyncio.StreamReader()
        if stderr_text:
            self.stderr.feed_data(stderr_text.encode("utf-8"))
        self.stderr.feed_eof()

        self._final_returncode = returncode
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = self._final_returncode
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        if self.returncode is None:
            self.returncode = -9


@pytest.mark.asyncio
async def test_exec_uses_app_server_jsonrpc_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    process = FakeProcess(
        [
            '{"id":1,"result":{"userAgent":"codex"}}',
            '{"id":2,"result":{"thread":{"id":"thread_1"}}}',
            '{"id":3,"result":{"turn":{"id":"turn_1"}}}',
            '{"method":"turn/started","params":{"threadId":"thread_1","turn":{"id":"turn_1","status":"inProgress"}}}',
            '{"method":"item/completed","params":{"threadId":"thread_1","turnId":"turn_1","item":{"type":"agentMessage","id":"item_1","text":"Hi!"}}}',
            '{"method":"thread/tokenUsage/updated","params":{"threadId":"thread_1","turnId":"turn_1","tokenUsage":{"last":{"inputTokens":1,"cachedInputTokens":0,"outputTokens":2}}}}',
            '{"method":"turn/completed","params":{"threadId":"thread_1","turn":{"id":"turn_1","status":"completed"}}}',
        ]
    )

    async def fake_create_subprocess_exec(*cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    exec_client = CodexExec(
        executable_path="codex-bin",
        env={"CUSTOM_ENV": "custom"},
        config_overrides={"approval_policy": "never"},
    )
    args = CodexExecArgs(
        input="hello",
        base_url="https://api.example.com",
        api_key="test-key",
        thread_id="thread_1",
        model="gpt-test",
        sandbox_mode="workspace-write",
        working_directory="/tmp/repo",
        additional_directories=["/tmp/shared-a"],
        model_reasoning_effort="high",
        network_access_enabled=True,
        web_search_mode="cached",
        approval_policy="on-request",
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )

    lines = [line async for line in exec_client.run(args)]
    events = [json.loads(line) for line in lines]

    assert [event["type"] for event in events] == [
        "turn.started",
        "item.completed",
        "turn.completed",
    ]
    assert events[-1]["usage"] == {
        "input_tokens": 1,
        "cached_input_tokens": 0,
        "output_tokens": 2,
    }

    cmd = list(captured["cmd"])  # type: ignore[arg-type]
    assert cmd[0] == "codex-bin"
    arg_list = cmd[1:]
    assert "app-server" in arg_list
    assert "exec" not in arg_list

    kwargs = captured["kwargs"]  # type: ignore[assignment]
    env = kwargs["env"]  # type: ignore[index]
    assert env["CUSTOM_ENV"] == "custom"  # type: ignore[index]
    assert env["OPENAI_BASE_URL"] == "https://api.example.com"  # type: ignore[index]
    assert env["CODEX_API_KEY"] == "test-key"  # type: ignore[index]
    assert env["CODEX_INTERNAL_ORIGINATOR_OVERRIDE"] == "codex_sdk_py"  # type: ignore[index]

    sent_messages = [json.loads(line) for line in process.stdin.data.decode("utf-8").splitlines()]
    assert sent_messages[0]["method"] == "initialize"
    assert sent_messages[1]["method"] == "initialized"
    assert sent_messages[2]["method"] == "thread/resume"
    assert sent_messages[3]["method"] == "turn/start"


@pytest.mark.asyncio
async def test_exec_emits_thread_started_once_for_thread_start_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(
        [
            '{"id":1,"result":{"userAgent":"codex"}}',
            '{"id":2,"result":{"thread":{"id":"thread_1"}}}',
            '{"id":3,"result":{"turn":{"id":"turn_1"}}}',
            '{"method":"thread/started","params":{"thread":{"id":"thread_1"}}}',
            '{"method":"turn/started","params":{"threadId":"thread_1","turn":{"id":"turn_1","status":"inProgress"}}}',
            '{"method":"turn/completed","params":{"threadId":"thread_1","turn":{"id":"turn_1","status":"completed"}}}',
        ]
    )

    async def fake_create_subprocess_exec(*_cmd, **_kwargs):  # noqa: ANN001
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    exec_client = CodexExec(executable_path="codex-bin")
    events = [json.loads(line) async for line in exec_client.run(CodexExecArgs(input="hello"))]
    event_types = [event["type"] for event in events]

    assert event_types == ["thread.started", "turn.started", "turn.completed"]
    assert event_types.count("thread.started") == 1


@pytest.mark.asyncio
async def test_exec_maps_file_change_in_progress_and_declined_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(
        [
            '{"id":1,"result":{"userAgent":"codex"}}',
            '{"id":2,"result":{"thread":{"id":"thread_1"}}}',
            '{"id":3,"result":{"turn":{"id":"turn_1"}}}',
            '{"method":"thread/started","params":{"thread":{"id":"thread_1"}}}',
            '{"method":"turn/started","params":{"threadId":"thread_1","turn":{"id":"turn_1","status":"inProgress"}}}',
            '{"method":"item/started","params":{"threadId":"thread_1","turnId":"turn_1","item":{"type":"fileChange","id":"item_1","status":"inProgress","changes":[{"path":"a.py","kind":"update"}]}}}',
            '{"method":"item/completed","params":{"threadId":"thread_1","turnId":"turn_1","item":{"type":"fileChange","id":"item_1","status":"declined","changes":[{"path":"a.py","kind":"update"}]}}}',
            '{"method":"turn/completed","params":{"threadId":"thread_1","turn":{"id":"turn_1","status":"completed"}}}',
        ]
    )

    async def fake_create_subprocess_exec(*_cmd, **_kwargs):  # noqa: ANN001
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    exec_client = CodexExec(executable_path="codex-bin")
    events = [json.loads(line) async for line in exec_client.run(CodexExecArgs(input="hello"))]

    started_item = next(
        event["item"] for event in events if event["type"] == "item.started"
    )
    completed_item = next(
        event["item"] for event in events if event["type"] == "item.completed"
    )

    assert started_item["type"] == "file_change"
    assert started_item["status"] == "in_progress"
    assert completed_item["type"] == "file_change"
    assert completed_item["status"] == "declined"


@pytest.mark.asyncio
async def test_exec_handles_dynamic_tool_request_with_failure_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(
        [
            '{"id":1,"result":{"userAgent":"codex"}}',
            '{"id":2,"result":{"thread":{"id":"thread_1"}}}',
            '{"id":3,"result":{"turn":{"id":"turn_1"}}}',
            '{"id":91,"method":"item/tool/call","params":{"threadId":"thread_1","turnId":"turn_1","callId":"call_1","tool":"demo","arguments":{}}}',
            '{"method":"turn/completed","params":{"threadId":"thread_1","turn":{"id":"turn_1","status":"completed"}}}',
        ]
    )

    async def fake_create_subprocess_exec(*_cmd, **_kwargs):  # noqa: ANN001
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    exec_client = CodexExec(executable_path="codex-bin")
    _ = [line async for line in exec_client.run(CodexExecArgs(input="hello", thread_id="thread_1"))]
    sent_messages = [json.loads(line) for line in process.stdin.data.decode("utf-8").splitlines()]

    dynamic_tool_response = next(message for message in sent_messages if message.get("id") == 91)
    assert dynamic_tool_response["result"]["success"] is False
    assert dynamic_tool_response["result"]["contentItems"][0]["type"] == "inputText"


@pytest.mark.asyncio
async def test_exec_uses_generic_error_for_unknown_server_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(
        [
            '{"id":1,"result":{"userAgent":"codex"}}',
            '{"id":2,"result":{"thread":{"id":"thread_1"}}}',
            '{"id":3,"result":{"turn":{"id":"turn_1"}}}',
            '{"id":77,"method":"item/unknown","params":{}}',
            '{"method":"turn/completed","params":{"threadId":"thread_1","turn":{"id":"turn_1","status":"completed"}}}',
        ]
    )

    async def fake_create_subprocess_exec(*_cmd, **_kwargs):  # noqa: ANN001
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    exec_client = CodexExec(executable_path="codex-bin")
    _ = [
        line
        async for line in exec_client.run(CodexExecArgs(input="hello", thread_id="thread_1"))
    ]
    sent_messages = [json.loads(line) for line in process.stdin.data.decode("utf-8").splitlines()]

    unknown_response = next(message for message in sent_messages if message.get("id") == 77)
    assert unknown_response["error"]["code"] == -32000
    assert "item/unknown" in unknown_response["error"]["message"]


@pytest.mark.asyncio
async def test_exec_rejects_skip_git_repo_check_in_app_server_mode() -> None:
    exec_client = CodexExec(executable_path="codex-bin")
    with pytest.raises(
        RuntimeError,
        match="skip_git_repo_check is not supported by the app-server transport",
    ):
        async for _ in exec_client.run(CodexExecArgs(input="hello", skip_git_repo_check=True)):
            pass


@pytest.mark.asyncio
async def test_exec_raises_on_rpc_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_create_subprocess_exec(*_cmd, **_kwargs):  # noqa: ANN001
        return FakeProcess(
            stdout_lines=['{"id":1,"error":{"code":-32000,"message":"not initialized"}}']
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    exec_client = CodexExec(executable_path="codex-bin")
    with pytest.raises(RuntimeError, match="not initialized"):
        async for _ in exec_client.run(CodexExecArgs(input="hello")):
            pass


def test_serialize_config_overrides() -> None:
    overrides = serialize_config_overrides(
        {
            "approval_policy": "never",
            "sandbox_workspace_write": {"network_access": True},
            "retry_budget": 3,
            "tool_rules": {"allow": ["git status", "git diff"]},
        }
    )

    assert 'approval_policy="never"' in overrides
    assert "sandbox_workspace_write.network_access=true" in overrides
    assert "retry_budget=3" in overrides
    assert 'tool_rules.allow=["git status", "git diff"]' in overrides
