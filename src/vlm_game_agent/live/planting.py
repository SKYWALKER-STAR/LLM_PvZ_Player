"""Validate and verify a single Gemini Live planting proposal."""

import time

from .observer import PlantProposal


class LivePlanter:
    def __init__(self, source, execute: bool):
        self.source = source
        self.execute = execute
        self.memory = None
        self.reader = None
        self.executor = None

    def place(self, proposal: PlantProposal) -> dict:
        try:
            return self._place(proposal)
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

    def _place(self, proposal: PlantProposal) -> dict:
        from vlm_game_agent.pvz import PvZExecutor, PvZMemory, PvZStateReader

        if self.memory is None:
            self.memory = PvZMemory()
            self.reader = PvZStateReader(self.memory)
        if not self.memory.is_connected() and not self.memory.connect():
            return {"status": "skipped", "reason": "game_memory_unavailable"}
        capture = self.source.capture
        if capture._window._hWnd != self.memory._hwnd:
            return {"status": "skipped", "reason": "capture_and_game_window_differ"}

        state = self.reader.read_state()
        if not state.in_battle or state.is_paused or state.scene != 0 or state.level_end_countdown > 0:
            return {"status": "skipped", "reason": "daytime_battle_required"}
        if proposal.card_index >= len(state.seeds):
            return {"status": "skipped", "reason": "invalid_card_slot"}
        seed = state.seeds[proposal.card_index]
        if seed.index != proposal.card_index or seed.plant_type not in (0, 1):
            return {"status": "skipped", "reason": "peashooter_or_sunflower_required"}
        if not seed.is_ready or state.sun < seed.sun_cost:
            return {"status": "skipped", "reason": "card_unavailable_or_insufficient_sun"}
        if any((plant.row, plant.col) == (proposal.row, proposal.col) for plant in state.plants):
            return {"status": "skipped", "reason": "tile_occupied"}
        if not self.execute:
            return {"status": "dry_run", "plant": seed.name}

        if self.executor is None:
            self.executor = PvZExecutor(self.memory, lambda: capture._window_box)
        if self.executor.injector is None:
            return {"status": "skipped", "reason": "game_executor_unavailable"}
        result = self.executor.execute("place_plant", proposal.model_dump(), state)
        if result.get("status") != "ok":
            return result
        time.sleep(0.3)
        updated = self.reader.read_state()
        verified = any(
            (plant.row, plant.col, plant.plant_type) == (proposal.row, proposal.col, seed.plant_type)
            for plant in updated.plants
        )
        return {**result, "status": "ok" if verified else "unverified", "verified": verified}

    def close(self) -> None:
        try:
            if self.executor is not None:
                self.executor.close()
        finally:
            if self.memory is not None:
                self.memory.disconnect()
