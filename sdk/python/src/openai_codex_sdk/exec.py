from __future__ import annotations

import asyncio
import json
import math
import os
import platform
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Mapping, Sequence

from .options import (
    ApprovalMode,
    CodexConfigObject,
    CodexConfigValue,
    ModelReasoningEffort,
    SandboxMode,
    WebSearchMode,
)

INTERNAL_ORIGINATOR_ENV = "CODEX_INTERNAL_ORIGINATOR_OVERRIDE"
PYTHON_SDK_ORIGINATOR = "codex_sdk_py"

_TOML_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(slots=True)
class CodexExecArgs:
    input: str
    base_url: str | None = None
    api_key: str | None = None
    thread_id: str | None = None
    images: Sequence[str] | None = None
    model: str | None = None
    sandbox_mode: SandboxMode | None = None
    working_directory: str | None = None
    additional_directories: Sequence[str] | None = None
    skip_git_repo_check: bool = False
    output_schema: dict[str, Any] | None = None
    model_reasoning_effort: ModelReasoningEffort | None = None
    signal: asyncio.Event | None = None
    network_access_enabled: bool | None = None
    web_search_mode: WebSearchMode | None = None
    web_search_enabled: bool | None = None
    approval_policy: ApprovalMode | None = None


@dataclass(slots=True)
class _UsageSnapshot:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


class CodexExec:
    def __init__(
        self,
        executable_path: str | None = None,
        env: dict[str, str] | None = None,
        config_overrides: CodexConfigObject | None = None,
    ) -> None:
        self._executable_path = executable_path or find_codex_path()
        self._env_override = env
        self._config_overrides = config_overrides

    async def run(self, args: CodexExecArgs) -> AsyncGenerator[str, None]:
        if args.signal is not None and args.signal.is_set():
            raise asyncio.CancelledError("Codex execution aborted before start")
        if args.skip_git_repo_check:
            raise RuntimeError(
                "skip_git_repo_check is not supported by the app-server transport."
            )

        command_args = self._build_command_args()
        env = self._build_env(args)

        process = await asyncio.create_subprocess_exec(
            self._executable_path,
            *command_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        if process.stdin is None:
            await _terminate_process(process)
            raise RuntimeError("Child process has no stdin")
        if process.stdout is None:
            await _terminate_process(process)
            raise RuntimeError("Child process has no stdout")

        stderr_chunks = bytearray()
        stderr_task = asyncio.create_task(_collect_stream(process.stderr, stderr_chunks))
        usage_by_turn: dict[tuple[str, str], _UsageSnapshot] = {}

        request_id = 1
        pending_events: list[dict[str, Any]] = []

        try:
            await _send_message(
                process.stdin,
                {
                    "id": request_id,
                    "method": "initialize",
                    "params": {
                        "clientInfo": {
                            "name": "codex_python_sdk",
                            "title": "Codex Python SDK",
                            "version": "0.0.0",
                        },
                        "capabilities": {"experimentalApi": True},
                    },
                },
            )
            await self._read_until_response(
                process,
                request_id,
                args.signal,
                pending_events,
                usage_by_turn,
            )
            request_id += 1

            await _send_message(process.stdin, {"method": "initialized"})

            thread_method, thread_params = self._build_thread_request(args)
            await _send_message(
                process.stdin,
                {
                    "id": request_id,
                    "method": thread_method,
                    "params": thread_params,
                },
            )
            thread_response = await self._read_until_response(
                process,
                request_id,
                args.signal,
                pending_events,
                usage_by_turn,
            )
            request_id += 1
            thread_id = _extract_thread_id(thread_response, thread_method)

            await _send_message(
                process.stdin,
                {
                    "id": request_id,
                    "method": "turn/start",
                    "params": self._build_turn_start_request(args, thread_id),
                },
            )
            turn_response = await self._read_until_response(
                process,
                request_id,
                args.signal,
                pending_events,
                usage_by_turn,
            )
            turn_id = _extract_turn_id(turn_response)

            while pending_events:
                yield json.dumps(pending_events.pop(0), separators=(",", ":"))

            turn_done = False
            while not turn_done:
                message = await _read_jsonrpc_message(process, args.signal)
                if _is_response(message) or _is_error(message):
                    continue
                if _is_request(message):
                    await self._handle_server_request(process.stdin, message)
                    continue
                if not _is_notification(message):
                    continue

                notification = _as_mapping(message)
                params = _as_mapping(notification.get("params"))
                method = str(notification.get("method", ""))

                turn_done |= _translate_notification_to_events(
                    method,
                    params,
                    pending_events,
                    usage_by_turn,
                    target_turn_id=turn_id,
                )

                while pending_events:
                    yield json.dumps(pending_events.pop(0), separators=(",", ":"))
        finally:
            if process.returncode is None:
                await _terminate_process(process)
            await stderr_task
            if process.returncode not in (0, -15) and process.returncode is not None:
                stderr_text = stderr_chunks.decode("utf-8")
                raise RuntimeError(
                    f"Codex app-server exited with code {process.returncode}: {stderr_text}"
                )

    async def _read_until_response(
        self,
        process: asyncio.subprocess.Process,
        request_id: int,
        signal: asyncio.Event | None,
        pending_events: list[dict[str, Any]],
        usage_by_turn: dict[tuple[str, str], _UsageSnapshot],
    ) -> Mapping[str, Any]:
        while True:
            message = await _read_jsonrpc_message(process, signal)
            if _is_response(message):
                response = _as_mapping(message)
                if _response_id_matches(response, request_id):
                    result = response.get("result")
                    if not isinstance(result, Mapping):
                        raise RuntimeError(f"Expected object result for request {request_id}")
                    return result
                continue

            if _is_error(message):
                error = _as_mapping(message)
                if _response_id_matches(error, request_id):
                    error_obj = _as_mapping(error.get("error"))
                    msg = str(error_obj.get("message", "Unknown app-server error"))
                    raise RuntimeError(msg)
                continue

            if _is_request(message):
                await self._handle_server_request(process.stdin, _as_mapping(message))
                continue

            if not _is_notification(message):
                continue

            notification = _as_mapping(message)
            params = _as_mapping(notification.get("params"))
            method = str(notification.get("method", ""))
            _translate_notification_to_events(
                method,
                params,
                pending_events,
                usage_by_turn,
                target_turn_id=None,
            )

    async def _handle_server_request(
        self,
        stdin: asyncio.StreamWriter,
        request: Mapping[str, Any],
    ) -> None:
        method = str(request.get("method", ""))
        request_id = request.get("id")

        if request_id is None:
            return

        if method == "item/commandExecution/requestApproval":
            await _send_message(
                stdin,
                {
                    "id": request_id,
                    "result": {"decision": "accept"},
                },
            )
            return

        if method == "item/fileChange/requestApproval":
            await _send_message(
                stdin,
                {
                    "id": request_id,
                    "result": {"decision": "accept"},
                },
            )
            return

        if method == "item/tool/requestUserInput":
            await _send_message(
                stdin,
                {
                    "id": request_id,
                    "result": {"answers": {}},
                },
            )
            return

        if method == "item/tool/call":
            await _send_message(
                stdin,
                {
                    "id": request_id,
                    "result": {
                        "contentItems": [
                            {
                                "type": "inputText",
                                "text": (
                                    "Dynamic tool calls are not supported by openai_codex_sdk "
                                    "yet."
                                ),
                            }
                        ],
                        "success": False,
                    },
                },
            )
            return

        if method == "account/chatgptAuthTokens/refresh":
            await _send_message(
                stdin,
                {
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": (
                            "Token refresh is not implemented in openai_codex_sdk. "
                            "Refresh credentials outside the SDK and retry."
                        ),
                    },
                },
            )
            return

        await _send_message(
            stdin,
            {
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": (
                        "Unsupported server request method for openai_codex_sdk: "
                        f"{method}"
                    ),
                },
            },
        )

    def _build_command_args(self) -> list[str]:
        command_args: list[str] = []
        if self._config_overrides:
            for override in serialize_config_overrides(self._config_overrides):
                command_args.extend(["--config", override])
        command_args.append("app-server")
        return command_args

    def _build_thread_request(self, args: CodexExecArgs) -> tuple[str, dict[str, Any]]:
        config = self._build_session_config(args)
        params: dict[str, Any] = {}

        if args.model:
            params["model"] = args.model
        if args.working_directory:
            params["cwd"] = args.working_directory
        if args.approval_policy:
            params["approvalPolicy"] = args.approval_policy
        if args.sandbox_mode:
            params["sandbox"] = args.sandbox_mode
        if config:
            params["config"] = config

        if args.thread_id:
            params["threadId"] = args.thread_id
            return "thread/resume", params

        return "thread/start", params

    def _build_turn_start_request(self, args: CodexExecArgs, thread_id: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": _build_turn_input(args.input, args.images),
        }

        if args.model:
            params["model"] = args.model
        if args.working_directory:
            params["cwd"] = args.working_directory
        if args.approval_policy:
            params["approvalPolicy"] = args.approval_policy
        if args.model_reasoning_effort:
            params["effort"] = args.model_reasoning_effort
        if args.output_schema is not None:
            params["outputSchema"] = args.output_schema

        sandbox_policy = _build_sandbox_policy(args)
        if sandbox_policy is not None:
            params["sandboxPolicy"] = sandbox_policy

        return params

    def _build_session_config(self, args: CodexExecArgs) -> dict[str, Any]:
        config: dict[str, Any] = {}

        if args.web_search_mode:
            config["web_search"] = args.web_search_mode
        elif args.web_search_enabled is True:
            config["web_search"] = "live"
        elif args.web_search_enabled is False:
            config["web_search"] = "disabled"

        return config

    def _build_env(self, args: CodexExecArgs) -> dict[str, str]:
        env: dict[str, str]
        if self._env_override is not None:
            env = dict(self._env_override)
        else:
            env = {k: v for k, v in os.environ.items()}

        if INTERNAL_ORIGINATOR_ENV not in env:
            env[INTERNAL_ORIGINATOR_ENV] = PYTHON_SDK_ORIGINATOR
        if args.base_url:
            env["OPENAI_BASE_URL"] = args.base_url
        if args.api_key:
            env["CODEX_API_KEY"] = args.api_key
        return env


async def _send_message(stdin: asyncio.StreamWriter, message: Mapping[str, Any]) -> None:
    encoded = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
    stdin.write(encoded)
    await stdin.drain()


def _build_turn_input(prompt: str, images: Sequence[str] | None) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = [{"type": "text", "text": prompt, "textElements": []}]
    if images:
        for image_path in images:
            input_items.append({"type": "localImage", "path": image_path})
    return input_items


def _build_sandbox_policy(args: CodexExecArgs) -> dict[str, Any] | None:
    has_workspace_fields = bool(args.additional_directories) or args.network_access_enabled is not None
    if args.sandbox_mode == "read-only":
        return {"type": "readOnly"}
    if args.sandbox_mode == "danger-full-access":
        return {"type": "dangerFullAccess"}
    if args.sandbox_mode == "workspace-write" or has_workspace_fields:
        roots: list[str] = []
        if args.working_directory:
            roots.append(str(Path(args.working_directory).resolve()))
        if args.additional_directories:
            roots.extend(str(Path(path).resolve()) for path in args.additional_directories)

        payload: dict[str, Any] = {"type": "workspaceWrite"}
        if roots:
            payload["writableRoots"] = roots
        if args.network_access_enabled is not None:
            payload["networkAccess"] = args.network_access_enabled
        return payload
    return None


def _translate_notification_to_events(
    method: str,
    params: Mapping[str, Any],
    pending_events: list[dict[str, Any]],
    usage_by_turn: dict[tuple[str, str], _UsageSnapshot],
    target_turn_id: str | None,
) -> bool:
    if method == "thread/tokenUsage/updated":
        _capture_token_usage(params, usage_by_turn)
        return False

    if method == "thread/started":
        thread = _as_mapping(params.get("thread"))
        thread_id = str(thread.get("id", ""))
        if thread_id:
            pending_events.append({"type": "thread.started", "thread_id": thread_id})
        return False

    if method == "turn/started":
        pending_events.append({"type": "turn.started"})
        return False

    if method == "item/started":
        item = _normalize_item(_as_mapping(params.get("item")))
        if item is not None:
            pending_events.append({"type": "item.started", "item": item})
        return False

    if method == "item/completed":
        item = _normalize_item(_as_mapping(params.get("item")))
        if item is not None:
            pending_events.append({"type": "item.completed", "item": item})
        return False

    if method == "item/agentMessage/delta":
        item_id = str(params.get("itemId", ""))
        pending_events.append(
            {
                "type": "item.updated",
                "item": {
                    "type": "agent_message",
                    "id": item_id,
                    "text": str(params.get("delta", "")),
                },
            }
        )
        return False

    if method == "error":
        error = _as_mapping(params.get("error"))
        message = str(error.get("message", "Unknown error"))
        pending_events.append({"type": "error", "message": message})
        return False

    if method == "turn/completed":
        thread_id = str(params.get("threadId", ""))
        turn = _as_mapping(params.get("turn"))
        turn_id = str(turn.get("id", ""))
        status = str(turn.get("status", ""))

        if status == "failed":
            turn_error = _as_mapping(turn.get("error"))
            message = str(turn_error.get("message", "Turn failed"))
            pending_events.append({"type": "turn.failed", "error": {"message": message}})
        elif status == "interrupted":
            pending_events.append(
                {
                    "type": "turn.failed",
                    "error": {"message": "Turn interrupted"},
                }
            )
        else:
            usage = usage_by_turn.pop((thread_id, turn_id), _UsageSnapshot(0, 0, 0))
            pending_events.append(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "cached_input_tokens": usage.cached_input_tokens,
                        "output_tokens": usage.output_tokens,
                    },
                }
            )

        return target_turn_id is None or turn_id == target_turn_id

    return False


def _capture_token_usage(
    params: Mapping[str, Any],
    usage_by_turn: dict[tuple[str, str], _UsageSnapshot],
) -> None:
    thread_id = str(params.get("threadId", ""))
    turn_id = str(params.get("turnId", ""))
    token_usage = _as_mapping(params.get("tokenUsage"))
    last = _as_mapping(token_usage.get("last"))

    if not thread_id or not turn_id:
        return

    usage_by_turn[(thread_id, turn_id)] = _UsageSnapshot(
        input_tokens=int(last.get("inputTokens", 0) or 0),
        cached_input_tokens=int(last.get("cachedInputTokens", 0) or 0),
        output_tokens=int(last.get("outputTokens", 0) or 0),
    )


def _normalize_item(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    item_type = str(raw.get("type", ""))

    if item_type == "agentMessage":
        return {
            "type": "agent_message",
            "id": str(raw.get("id", "")),
            "text": str(raw.get("text", "")),
        }

    if item_type == "reasoning":
        summary = raw.get("summary")
        content = raw.get("content")
        summary_parts = [str(part) for part in summary] if isinstance(summary, list) else []
        content_parts = [str(part) for part in content] if isinstance(content, list) else []
        text = "\n".join(summary_parts + content_parts)
        return {
            "type": "reasoning",
            "id": str(raw.get("id", "")),
            "text": text,
        }

    if item_type == "commandExecution":
        return {
            "type": "command_execution",
            "id": str(raw.get("id", "")),
            "command": str(raw.get("command", "")),
            "aggregated_output": str(raw.get("aggregatedOutput", "")),
            "status": _normalize_command_status(str(raw.get("status", ""))),
            "exit_code": _optional_int(raw.get("exitCode")),
        }

    if item_type == "fileChange":
        changes: list[dict[str, str]] = []
        raw_changes = raw.get("changes")
        if isinstance(raw_changes, list):
            for change in raw_changes:
                if not isinstance(change, Mapping):
                    continue
                kind_payload = change.get("kind")
                kind_type = "update"
                if isinstance(kind_payload, Mapping):
                    kind_type = str(kind_payload.get("type", "update"))
                elif isinstance(kind_payload, str):
                    kind_type = kind_payload
                changes.append(
                    {
                        "path": str(change.get("path", "")),
                        "kind": _normalize_patch_kind(kind_type),
                    }
                )

        return {
            "type": "file_change",
            "id": str(raw.get("id", "")),
            "changes": changes,
            "status": _normalize_patch_status(str(raw.get("status", ""))),
        }

    if item_type == "mcpToolCall":
        normalized: dict[str, Any] = {
            "type": "mcp_tool_call",
            "id": str(raw.get("id", "")),
            "server": str(raw.get("server", "")),
            "tool": str(raw.get("tool", "")),
            "arguments": raw.get("arguments"),
            "status": _normalize_mcp_status(str(raw.get("status", ""))),
        }

        result = raw.get("result")
        if isinstance(result, Mapping):
            normalized["result"] = {
                "content": list(result.get("content", []))
                if isinstance(result.get("content"), list)
                else [],
                "structured_content": result.get("structuredContent"),
            }

        error = raw.get("error")
        if isinstance(error, Mapping):
            normalized["error"] = {"message": str(error.get("message", ""))}

        return normalized

    if item_type == "webSearch":
        return {
            "type": "web_search",
            "id": str(raw.get("id", "")),
            "query": str(raw.get("query", "")),
        }

    return None


def _normalize_command_status(status: str) -> str:
    if status == "inProgress":
        return "in_progress"
    if status in {"completed", "failed", "declined"}:
        return status
    return "in_progress"


def _normalize_patch_status(status: str) -> str:
    if status == "inProgress":
        return "in_progress"
    if status in {"completed", "failed", "declined"}:
        return status
    return "in_progress"


def _normalize_patch_kind(kind: str) -> str:
    if kind in {"add", "delete", "update"}:
        return kind
    return "update"


def _normalize_mcp_status(status: str) -> str:
    if status == "inProgress":
        return "in_progress"
    if status in {"completed", "failed"}:
        return status
    return "in_progress"


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _extract_thread_id(result: Mapping[str, Any], method: str) -> str:
    thread = _as_mapping(result.get("thread"))
    thread_id = str(thread.get("id", ""))
    if not thread_id:
        raise RuntimeError(f"{method} response missing thread.id")
    return thread_id


def _extract_turn_id(result: Mapping[str, Any]) -> str:
    turn = _as_mapping(result.get("turn"))
    turn_id = str(turn.get("id", ""))
    if not turn_id:
        raise RuntimeError("turn/start response missing turn.id")
    return turn_id


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _is_request(message: Mapping[str, Any]) -> bool:
    return "method" in message and "id" in message


def _is_notification(message: Mapping[str, Any]) -> bool:
    return "method" in message and "id" not in message


def _is_response(message: Mapping[str, Any]) -> bool:
    return "id" in message and "result" in message and "method" not in message


def _is_error(message: Mapping[str, Any]) -> bool:
    return "id" in message and "error" in message and "method" not in message


def _response_id_matches(message: Mapping[str, Any], request_id: int) -> bool:
    response_id = message.get("id")
    if isinstance(response_id, int):
        return response_id == request_id
    if isinstance(response_id, str):
        return response_id == str(request_id)
    return False


async def _read_jsonrpc_message(
    process: asyncio.subprocess.Process,
    signal: asyncio.Event | None,
) -> Mapping[str, Any]:
    if process.stdout is None:
        raise RuntimeError("Child process has no stdout")

    while True:
        line = await _readline_with_signal(process.stdout, process, signal)
        if line == b"":
            raise RuntimeError("Codex app-server closed stdout unexpectedly")

        text = line.decode("utf-8").strip()
        if not text:
            continue

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, Mapping):
            return parsed


async def _collect_stream(
    stream: asyncio.StreamReader | None,
    into: bytearray,
) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return
        into.extend(chunk)


async def _readline_with_signal(
    stream: asyncio.StreamReader,
    process: asyncio.subprocess.Process,
    signal: asyncio.Event | None,
) -> bytes:
    if signal is None:
        return await stream.readline()

    read_task = asyncio.create_task(stream.readline())
    signal_task = asyncio.create_task(signal.wait())
    done, pending = await asyncio.wait(
        {read_task, signal_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    if signal_task in done:
        read_task.cancel()
        await asyncio.gather(read_task, return_exceptions=True)
        await _terminate_process(process)
        raise asyncio.CancelledError("Codex execution aborted")
    signal_task.cancel()
    return await read_task


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


def serialize_config_overrides(config_overrides: CodexConfigObject) -> list[str]:
    overrides: list[str] = []
    flatten_config_overrides(config_overrides, "", overrides)
    return overrides


def flatten_config_overrides(
    value: CodexConfigValue,
    prefix: str,
    overrides: list[str],
) -> None:
    if not _is_plain_object(value):
        if prefix:
            overrides.append(f"{prefix}={to_toml_value(value, prefix)}")
            return
        raise ValueError("Codex config overrides must be a plain object")

    entries = list(value.items())
    if not prefix and len(entries) == 0:
        return

    if prefix and len(entries) == 0:
        overrides.append(f"{prefix}={{}}")
        return

    for key, child in entries:
        if key == "":
            raise ValueError("Codex config override keys must be non-empty strings")
        path = f"{prefix}.{key}" if prefix else key
        if _is_plain_object(child):
            flatten_config_overrides(child, path, overrides)
        else:
            overrides.append(f"{path}={to_toml_value(child, path)}")


def to_toml_value(value: CodexConfigValue, path: str) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Codex config override at {path} must be a finite number")
        return str(value)
    if isinstance(value, list):
        rendered = [to_toml_value(item, f"{path}[{idx}]") for idx, item in enumerate(value)]
        return f"[{', '.join(rendered)}]"
    if _is_plain_object(value):
        parts: list[str] = []
        for key, child in value.items():
            if key == "":
                raise ValueError("Codex config override keys must be non-empty strings")
            parts.append(f"{_format_toml_key(key)} = {to_toml_value(child, f'{path}.{key}')}")
        return "{" + ", ".join(parts) + "}"
    raise ValueError(f"Unsupported Codex config override value at {path}: {type(value).__name__}")


def _format_toml_key(key: str) -> str:
    if _TOML_BARE_KEY.fullmatch(key):
        return key
    return json.dumps(key)


def _is_plain_object(value: object) -> bool:
    return isinstance(value, Mapping)


def find_codex_path() -> str:
    target_triple = _platform_target_triple()
    script_dir = Path(__file__).resolve().parent
    vendor_root = script_dir.parent / "vendor"
    binary_name = "codex.exe" if os.name == "nt" else "codex"
    binary_path = vendor_root / target_triple / "codex" / binary_name
    if binary_path.exists():
        return str(binary_path)

    on_path = shutil.which("codex")
    if on_path:
        return on_path
    raise RuntimeError(
        "Unable to locate codex binary. "
        "Set codex_path_override or ensure 'codex' is available on PATH."
    )


def _platform_target_triple() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        if machine in {"x86_64", "amd64"}:
            return "x86_64-unknown-linux-musl"
        if machine in {"arm64", "aarch64"}:
            return "aarch64-unknown-linux-musl"
    if system == "darwin":
        if machine in {"x86_64", "amd64"}:
            return "x86_64-apple-darwin"
        if machine in {"arm64", "aarch64"}:
            return "aarch64-apple-darwin"
    if system == "windows":
        if machine in {"x86_64", "amd64"}:
            return "x86_64-pc-windows-msvc"
        if machine in {"arm64", "aarch64"}:
            return "aarch64-pc-windows-msvc"
    raise RuntimeError(f"Unsupported platform: {system} ({machine})")
