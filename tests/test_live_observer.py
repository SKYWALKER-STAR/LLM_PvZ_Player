import asyncio
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_game_agent.live.observer import LiveObserver
from vlm_game_agent.live.planting import LivePlanter


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
    async def test_live_proposal_is_executed_and_verified_once(self):
        source = NS(capture=NS(_window=NS(_hWnd=10)))
        planter = LivePlanter(source, execute=True)
        planter.memory = NS(_hwnd=10, is_connected=lambda: True)
        seed = NS(index=0, plant_type=0, is_ready=True, sun_cost=100, name="豌豆射手")
        state = NS(in_battle=True, is_paused=False, scene=0, level_end_countdown=0,
                   seeds=[seed], sun=100, plants=[])
        planter.reader = NS(read_state=Mock(side_effect=[state, NS(plants=[NS(row=0, col=2, plant_type=0)])]))
        planter.executor = NS(injector=object(), execute=Mock(return_value={"status": "ok"}))
        observer = LiveObserver(source, Path("unused"), timeout=5, planter=planter)
        observer.pending = (1, time.monotonic())
        records = []
        observer.record = records.append
        proposal = call()
        proposal.args["plant"] = {"card_index": 0, "row": 0, "col": 2}

        class Session:
            async def receive(self):
                yield message([proposal, proposal])
                yield message(complete=True)
                yield message(complete=True)
                raise asyncio.CancelledError

            async def send_tool_response(self, function_responses):
                pass

        with patch.dict(sys.modules, {"vlm_game_agent.pvz": NS(PvZExecutor=None, PvZMemory=None, PvZStateReader=None)}):
            with patch("vlm_game_agent.live.planting.time.sleep"), self.assertRaises(asyncio.CancelledError):
                await observer.receive(Session(), NS(FunctionResponse=lambda **kw: NS(**kw)))

        planter.executor.execute.assert_called_once_with("place_plant", proposal.args["plant"], state)
        self.assertEqual([event["type"] for event in records], ["observation", "plant_result"])
        self.assertTrue(records[1]["verified"])

    async def test_expired_live_proposal_is_discarded(self):
        planter = Mock()
        observer = LiveObserver(None, Path("unused"), timeout=5, planter=planter)
        observer.pending = (1, time.monotonic() - 6)
        records = []
        observer.record = records.append
        proposal = call()
        proposal.args["plant"] = {"card_index": 0, "row": 0, "col": 2}

        class Session:
            async def receive(self):
                yield message([proposal])
                yield message(complete=True)
                raise asyncio.CancelledError

            async def send_tool_response(self, function_responses):
                pass

        with self.assertRaises(asyncio.CancelledError):
            await observer.receive(Session(), NS(FunctionResponse=lambda **kw: NS(**kw)))
        planter.place.assert_not_called()
        self.assertEqual(records[-1]["reason"], "observation_expired")

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
