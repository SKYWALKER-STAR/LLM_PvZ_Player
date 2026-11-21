"""Gemini Live backend for the synchronous GameAgent decision loop."""

from __future__ import annotations

import asyncio
import base64
from typing import Any


class GeminiLiveAgentClient:
    last_prompt_tokens = 0

    def __init__(self, api_key: str, model: str, timeout: float = 30):
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
        self._types = types
        self._model = model
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._connection = None
        self._session = None
        self._instruction = ""
        self._history: list[tuple[str, str]] = []

    def chat(self, messages: list[dict[str, Any]]) -> tuple[str, str]:
        try:
            return self._loop.run_until_complete(asyncio.wait_for(self._decide(messages), self._timeout)), ""
        except Exception as exc:
            self._loop.run_until_complete(self._disconnect())
            code = getattr(exc, "code", None)
            status = f", code={code}" if isinstance(code, int) else ""
            raise RuntimeError(f"Gemini Live 决策失败 ({type(exc).__name__}{status})") from None

    async def _decide(self, messages: list[dict[str, Any]]) -> str:
        instruction = "\n".join(msg["content"] for msg in messages if msg["role"] == "system")
        history = []
        frame = None
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                continue
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        header, data = part["image_url"]["url"].split(",", 1)
                        frame = self._types.Blob(data=base64.b64decode(data), mime_type=header[5:].split(";")[0])
                content = "\n".join(part["text"] for part in content if part.get("type") == "text")
            if content.strip():
                history.append((role, content))

        if self._session is None or instruction != self._instruction:
            await self._disconnect()
            await self._connect(instruction)

        # History is bounded and may be trimmed by GameAgent. Keep the overlapping suffix.
        overlap = min(len(history), len(self._history))
        while overlap and self._history[-overlap:] != history[:overlap]:
            overlap -= 1
        new_history = history[overlap:]
        if frame is not None:
            await self._session.send_realtime_input(video=frame)
        context = "\n\n".join(f"[{role}]\n{text}" for role, text in new_history)
        await self._session.send_client_content(
            turns={"role": "user", "parts": [{"text": (
                f"{context}\n\n请调用 submit_decision 提交本轮决策。"
            )}]}, turn_complete=True,
        )

        decision = None
        async for message in self._session.receive():
            if message.go_away:
                raise ConnectionError("Live session is closing")
            if message.tool_call:
                responses = []
                for call in message.tool_call.function_calls:
                    output = (call.args or {}).get("output")
                    accepted = call.name == "submit_decision" and decision is None and isinstance(output, str) and bool(output.strip())
                    if accepted:
                        decision = output
                    responses.append(self._types.FunctionResponse(id=call.id, name=call.name, response={"ok": accepted}))
                await self._session.send_tool_response(function_responses=responses)
            if message.server_content and message.server_content.turn_complete and decision is not None:
                self._history = history
                return decision
        raise RuntimeError("Gemini Live 未返回完整决策")

    async def _connect(self, instruction: str) -> None:
        connection = self._client.aio.live.connect(model=self._model, config={
            "response_modalities": ["AUDIO"],
            "system_instruction": instruction + (
                "\n每轮只调用一次 submit_decision。把原约定格式的完整文本决定放入 output，"
                "包括需要的 <tool_call> 或 <compact> 块。调用后结束本轮，不要口头播报。"
            ),
            "tools": [{"function_declarations": [{
                "name": "submit_decision",
                "description": "提交本轮操作方案给游戏代理",
                "parameters": {"type": "OBJECT", "properties": {"output": {"type": "STRING"}}, "required": ["output"]},
            }]}],
            "context_window_compression": {"sliding_window": {}},
        })
        self._session = await connection.__aenter__()
        self._connection = connection
        self._instruction = instruction

    async def _disconnect(self) -> None:
        connection = self._connection
        self._connection = self._session = None
        self._history = []
        if connection is not None:
            try:
                await asyncio.wait_for(connection.__aexit__(None, None, None), 3)
            except Exception:
                pass

    def close(self) -> None:
        if self._loop.is_closed():
            return
        try:
            self._loop.run_until_complete(self._disconnect())
            self._loop.run_until_complete(self._client.aio.aclose())
        finally:
            self._loop.close()
