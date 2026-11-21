"""Stream screenshots to Gemini without pausing or controlling the game."""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: int = Field(ge=1)
    phase: str
    summary: str
    threats: list[str]
    suggestions: list[str]
    uncertainties: list[str]


SYSTEM_PROMPT = """你是植物大战僵尸的画面观察员。只观察，绝不操作游戏。
收到判断请求后，结合最新画面调用 report_game_state 一次，使用请求指定的 observation_id。
用中文描述场景阶段、局势、各行威胁、建议和无法确定的信息；不要猜测看不清的数字。
建议只是文本，不执行。仅在收到新的判断请求时报告，不需要口头解说。
工具返回成功后结束本轮，等待下一次请求。"""


class WindowSource:
    """Create/use/close the mss instance on one dedicated worker thread."""

    def __init__(self, title: str, scale: float = 1.0):
        self.title = title
        self.scale = scale
        self.capture = None

    def read(self) -> bytes:
        if self.capture is None:
            from vlm_game_agent.vision import CaptureConfig, WindowCapture

            self.capture = WindowCapture(CaptureConfig(
                output_format="JPEG", jpeg_quality=80, scale=self.scale,
                ensure_foreground=False, capture_area="client",
            ))
            self.capture.find_window(self.title)
        return self.capture.capture_to_bytes()

    def close(self):
        if self.capture is not None:
            self.capture.close()


class LiveObserver:
    def __init__(self, source, output: Path, interval: float = 5, timeout: float = 30):
        self.source = source
        self.output = output
        self.interval = interval
        self.timeout = timeout
        self.pending = None
        self.next_request = 0.0
        self.sequence = 0
        self.resume_handle = None

    def record(self, event: dict):
        event = {"time": datetime.now(timezone.utc).isoformat(), **event}
        line = json.dumps(event, ensure_ascii=False)
        print(line, flush=True)
        with self.output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    async def send_frames(self, session, types, pool):
        loop = asyncio.get_running_loop()
        while True:
            started = time.monotonic()
            # No unbounded frame queue: capture only when ready to send.
            frame = await loop.run_in_executor(pool, self.source.read)
            await session.send_realtime_input(video=types.Blob(data=frame, mime_type="image/jpeg"))
            now = time.monotonic()
            if self.pending and now - self.pending[1] > self.timeout:
                raise TimeoutError("Gemini 判断超时；重建会话")
            if self.pending is None and now >= self.next_request:
                self.sequence += 1
                self.pending = (self.sequence, now)
                await session.send_client_content(
                    turns={"role": "user", "parts": [{"text": (
                        f"请判断最新游戏画面，调用 report_game_state，observation_id={self.sequence}。"
                    )}]}, turn_complete=True,
                )
            await asyncio.sleep(max(0, 1 - (time.monotonic() - started)))

    async def receive(self, session, types):
        # SDK receive() ends at a turn boundary, not necessarily session end.
        while True:
            received = False
            async for message in session.receive():
                received = True
                if message.session_resumption_update:
                    update = message.session_resumption_update
                    if update.resumable and update.new_handle:
                        self.resume_handle = update.new_handle
                if message.go_away:
                    raise ConnectionError("服务端请求重连")
                if message.tool_call:
                    responses = []
                    for call in message.tool_call.function_calls:
                        try:
                            if call.name != "report_game_state":
                                raise ValueError("未知工具")
                            report = Observation.model_validate(call.args)
                            if (not self.pending or report.observation_id != self.pending[0]
                                    or getattr(self, "report_received", False)):
                                raise ValueError("过期或重复的判断")
                            self.record({"type": "observation", **report.model_dump(),
                                         "latency_seconds": round(time.monotonic() - self.pending[1], 3)})
                            # Wait for turn_complete before triggering another request.
                            self.report_received = True
                            result = {"ok": True}
                        except ValueError as exc:
                            result = {"ok": False, "error": str(exc)}
                        responses.append(types.FunctionResponse(id=call.id, name=call.name, response=result))
                    await session.send_tool_response(function_responses=responses)
                content = message.server_content
                if content and content.turn_complete:
                    if self.pending and not getattr(self, "report_received", False):
                        self.record({"type": "warning", "message": "本轮未返回有效结构化判断"})
                    self.pending = None
                    self.report_received = False
                    self.next_request = time.monotonic() + self.interval
            if not received:
                raise ConnectionError("Live 接收流已关闭")

    async def run(self, api_key: str, model: str):
        from google import genai
        from google.genai import types

        self.output.parent.mkdir(parents=True, exist_ok=True)
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvz-capture")
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
        try:
            # Fail fast for missing window/display before connecting to a paid API.
            await asyncio.get_running_loop().run_in_executor(pool, self.source.read)
            for attempt in range(4):
                config = {
                    "response_modalities": ["AUDIO"],
                    "system_instruction": SYSTEM_PROMPT,
                    "tools": [{"function_declarations": [{
                        "name": "report_game_state", "description": "报告游戏画面判断，不执行操作",
                        "parameters_json_schema": Observation.model_json_schema(),
                    }]}],
                    "context_window_compression": {"sliding_window": {}},
                    "session_resumption": {"handle": self.resume_handle},
                }
                self.pending = None
                self.report_received = False
                try:
                    async with client.aio.live.connect(model=model, config=config) as session:
                        self.record({"type": "connected", "model": model})
                        tasks = [asyncio.create_task(self.send_frames(session, types, pool)),
                                 asyncio.create_task(self.receive(session, types))]
                        try:
                            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                            for task in done:
                                task.result()
                        finally:
                            for task in tasks:
                                task.cancel()
                            await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as exc:
                    # Do not log SDK exception bodies, which may include credentials.
                    self.record({"type": "connection_error", "error_type": type(exc).__name__,
                                 "retry": attempt < 3})
                    if attempt == 3:
                        raise RuntimeError("Live 连接失败；请检查模型权限、API Key、网络及游戏窗口") from None
                    if isinstance(exc, TimeoutError):
                        self.resume_handle = None
                    await asyncio.sleep(2 ** attempt)
        finally:
            await asyncio.get_running_loop().run_in_executor(pool, self.source.close)
            pool.shutdown(wait=True)
            await client.aio.aclose()
