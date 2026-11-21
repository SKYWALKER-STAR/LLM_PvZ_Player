import asyncio
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_game_agent.live.observer import LiveObserver


def message(calls=None, complete=False):
    return NS(session_resumption_update=None, go_away=None,
              tool_call=NS(function_calls=calls) if calls else None,
              server_content=NS(turn_complete=complete))


def call(observation_id=1):
    return NS(name="report_game_state", id="tool-1", args={
        "observation_id": observation_id, "phase": "战斗", "summary": "第一行有僵尸",
        "threats": ["第一行"], "suggestions": ["补充防御"], "uncertainties": [],
    })


class ObserverTests(unittest.IsolatedAsyncioTestCase):
    async def test_reports_reject_stale_duplicate_and_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            observer = LiveObserver(None, Path(directory) / "out.jsonl")
            observer.pending = (1, time.monotonic())
            records = []
            observer.record = records.append
            invalid = call()
            invalid.args = {"observation_id": 1}
            calls = [call(9), invalid, call(), call()]
            responses = []

            class Session:
                count = 0

                async def receive(self):
                    self.count += 1
                    if self.count == 1:
                        yield message(calls)
                        yield message(complete=True)
                    else:
                        # Prove receive() is restarted after a completed turn.
                        raise asyncio.CancelledError

                async def send_tool_response(self, function_responses):
                    responses.extend(function_responses)

            session = Session()
            with self.assertRaises(asyncio.CancelledError):
                await observer.receive(session, NS(FunctionResponse=lambda **kw: NS(**kw)))
            self.assertEqual([r.response["ok"] for r in responses], [False, False, True, False])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["summary"], "第一行有僵尸")
            self.assertIsNone(observer.pending)
            self.assertEqual(session.count, 2)

    async def test_frame_sent_before_request_and_timeout(self):
        from concurrent.futures import ThreadPoolExecutor

        observer = LiveObserver(NS(read=lambda: b"jpeg"), Path("unused"), timeout=0.01)
        events = []

        class Session:
            async def send_realtime_input(self, **kw):
                events.append("frame")

            async def send_client_content(self, **kw):
                events.append("request")
                # Simulate a stalled decision without sleeping for the configured timeout.
                observer.pending = (1, time.monotonic() - 100)

        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(observer.send_frames(Session(), NS(Blob=lambda **kw: kw), pool), 3)
        self.assertEqual(events, ["frame", "request", "frame"])


if __name__ == "__main__":
    unittest.main()
