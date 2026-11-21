import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_game_agent.agent.parser import parse_tool_calls
from vlm_game_agent.live.agent_client import GeminiLiveAgentClient


class LiveAgentClientTests(unittest.TestCase):
    def test_decisions_reuse_session_and_reconnect_with_context(self):
        output = '<tool_call>{"name":"computer_use","arguments":{"action":"place_plant"}}</tool_call>'
        events = []

        class Session:
            broken = False

            async def send_realtime_input(self, *, video):
                events.append(("video", video.data))

            async def send_client_content(self, *, turns, turn_complete):
                events.append(("text", turns["parts"][0]["text"]))

            async def send_tool_response(self, *, function_responses):
                events.append(("ack", function_responses[0].response))

            async def receive(self):
                call = NS(name="submit_decision", id="decision", args={"output": output})
                yield NS(go_away=False, tool_call=NS(function_calls=[call]), server_content=None)
                if self.broken:
                    raise ConnectionError
                yield NS(go_away=False, tool_call=None, server_content=NS(turn_complete=True))

        first, second = Session(), Session()
        connect = Mock(side_effect=[
            NS(__aenter__=AsyncMock(return_value=session), __aexit__=AsyncMock())
            for session in (first, second)
        ])
        sdk = NS(Client=lambda **kwargs: NS(aio=NS(live=NS(connect=connect), aclose=AsyncMock())),
                 types=NS(Blob=NS, FunctionResponse=NS))
        initial = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        frame = {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,Zm9v"}},
            {"type": "text", "text": "state"},
        ]}
        with patch.dict(sys.modules, {"google": NS(genai=sdk), "google.genai": sdk}):
            client = GeminiLiveAgentClient("test-key", "test-model")
            self.addCleanup(client.close)
            answer, _ = client.chat([*initial, frame])
            self.assertEqual(parse_tool_calls(answer)[0].arguments["action"], "place_plant")
            self.assertEqual(events[0], ("video", b"foo"))
            feedback = {"role": "user", "content": "execution result"}
            client.chat([*initial, feedback, frame])
            self.assertEqual(connect.call_count, 1)
            self.assertIn("execution result", events[-2][1])
            first.broken = True
            with self.assertRaisesRegex(RuntimeError, "ConnectionError"):
                client.chat([*initial, feedback, frame])
            client.chat([*initial, feedback, frame])
            self.assertEqual(connect.call_count, 2)
            self.assertIn("task", events[-2][1])
