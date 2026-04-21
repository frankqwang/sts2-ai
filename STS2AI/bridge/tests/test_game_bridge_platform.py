from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_python_root = Path(__file__).resolve().parents[1]
if str(_python_root) not in sys.path:
    sys.path.insert(0, str(_python_root))

import game_bridge
from game_bridge.session.base import SessionFactory
from game_bridge.session.combat import CombatSession
from game_bridge.session.full_run import PipeBackedFullRunClient, create_full_run_client
from game_bridge.session.pool import SessionPool
from game_bridge.spectate.controller import SpectatorController
from game_bridge.spectate.overlay import OverlayWriter
from game_bridge.spectate.policy import NullPolicy, ReplayPolicy
from game_bridge.types import PolicyContext, SessionConfig


class _FakeSession:
    def __init__(self, terminal_after_act: bool = True):
        self.closed = False
        self._state = {
            "state_type": "map",
            "terminal": False,
            "run_outcome": None,
            "legal_actions": [{"action": "proceed", "is_enabled": True}],
        }
        self._terminal_after_act = terminal_after_act

    def reset(self, **_kwargs):
        return dict(self._state)

    def get_state(self):
        return dict(self._state)

    def act(self, _action):
        self._state = {
            "state_type": "game_over",
            "terminal": self._terminal_after_act,
            "run_outcome": "victory" if self._terminal_after_act else None,
            "legal_actions": [],
        }
        return dict(self._state)

    def close(self):
        self.closed = True


def test_session_factory_dispatch(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_combat(**kwargs):
        calls.append(("combat", kwargs))
        return "combat-session"

    def _fake_full_run(**kwargs):
        calls.append(("full_run", kwargs))
        return "full-run-session"

    monkeypatch.setattr("game_bridge.session.create_combat_session", _fake_combat)
    monkeypatch.setattr("game_bridge.session.create_full_run_session", _fake_full_run)

    combat_factory = SessionFactory(kind="combat", config=SessionConfig(port=2222, auto_launch=True))
    full_run_factory = SessionFactory(
        kind="full_run",
        config=SessionConfig(port=3333, use_pipe=True, transport="proto", base_url="http://localhost:9"),
    )

    assert combat_factory.create() == "combat-session"
    assert full_run_factory.create() == "full-run-session"
    assert calls[0][0] == "combat"
    assert calls[0][1]["port"] == 2222
    assert calls[1][0] == "full_run"
    assert calls[1][1]["port"] == 3333
    assert calls[1][1]["transport"] == "proto"


def test_session_pool_reuses_and_closes():
    created: list[_FakeSession] = []

    class _Factory:
        def create(self):
            session = _FakeSession()
            created.append(session)
            return session

    pool = SessionPool(factory=_Factory(), size=2)
    pool.warmup()

    assert len(created) == 2
    assert pool.get(0) is created[0]
    assert pool.get(2) is created[0]
    assert pool.get(3) is created[1]

    pool.close_all()
    assert all(session.closed for session in created)


def test_replay_policy_from_jsonl(tmp_path: Path):
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(
        json.dumps({"action": "a"}) + "\n" + json.dumps({"action": "b"}) + "\n",
        encoding="utf-8",
    )

    policy = ReplayPolicy.from_jsonl(replay_path)
    context = PolicyContext(step_index=0, character_id="IRONCLAD")

    assert policy.select_action({}, [], context) == {"action": "a"}
    assert policy.select_action({}, [], context) == {"action": "b"}
    assert policy.select_action({}, [], context) is None


def test_spectator_controller_writes_overlay(tmp_path: Path):
    session = _FakeSession()

    class _FirstActionPolicy:
        def select_action(self, _state, legal_actions, _context):
            return dict(legal_actions[0])

    overlay_path = tmp_path / "overlay.json"
    controller = SpectatorController(
        session=session,
        policy=_FirstActionPolicy(),
        overlay=OverlayWriter(overlay_path),
    )

    result = controller.play_episode(max_steps=5)

    assert result["terminal"] is True
    assert result["run_outcome"] == "victory"
    assert result["steps"] == 1
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert overlay["step_index"] == 0
    assert overlay["chosen_action"]["action"] == "proceed"


def test_null_policy_stops_without_action():
    session = _FakeSession(terminal_after_act=False)
    controller = SpectatorController(session=session, policy=NullPolicy())

    result = controller.play_episode(max_steps=5)

    assert result["stopped"] is True
    assert result["steps"] == 0


def test_game_bridge_source_has_no_legacy_training_imports():
    runtime_root = _python_root / "game_bridge"
    forbidden = ("networkV2.s5_net", "networkV2.s6_training", "archive.legacy_research")
    for path in runtime_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, f"{path} still references {marker}"


def test_full_run_auto_launch_defaults_are_resolved():
    client = PipeBackedFullRunClient(port=17777, protocol="proto", auto_launch=True)
    try:
        assert client.repo_root is not None
        assert client.host_path is not None
        assert client._conn.cfg.auto_launch is True
        assert client._conn.cfg.sim_launcher is not None
    finally:
        client.close()


def test_create_full_run_client_rejects_auto_launch_without_pipe():
    with pytest.raises(ValueError, match="auto_launch requires use_pipe=True"):
        create_full_run_client(auto_launch=True, use_pipe=False)


def test_combat_session_snapshot_methods_reuse_pipe_mixin():
    session = object.__new__(CombatSession)
    calls: list[tuple[str, dict[str, object] | None]] = []

    def _fake_call(method: str, params: dict[str, object] | None = None):
        calls.append((method, params))
        if method == "save_state":
            return {"state_id": "combat-state-1"}
        if method == "load_state":
            return {"state_type": "monster"}
        if method == "export_state":
            return {"path": str(params["path"])}
        if method == "import_state":
            return {"state_type": "monster"}
        if method == "delete_state":
            return {"deleted": True}
        raise AssertionError(f"unexpected method: {method}")

    session._call = _fake_call

    assert session.save_state() == "combat-state-1"
    assert session.load_state("combat-state-1")["state_type"] == "monster"
    assert session.export_state("C:/tmp/state.json") == "C:/tmp/state.json"
    assert session.import_state("C:/tmp/state.json")["state_type"] == "monster"
    assert session.delete_state("combat-state-1") is True
    assert [name for name, _ in calls] == [
        "save_state",
        "load_state",
        "export_state",
        "import_state",
        "delete_state",
    ]


def test_combat_session_call_allows_snapshot_rpcs():
    class _FakeConn:
        def __init__(self):
            self.connected = False
            self.calls: list[tuple[str, dict[str, object] | None]] = []

        def is_connected(self):
            return self.connected

        def connect(self):
            self.connected = True

        def safe_call(self, method, params):
            self.calls.append((method, params))
            return {"state_id": "snap-1"}

    session = object.__new__(CombatSession)
    session._conn = _FakeConn()

    result = CombatSession._call(session, "save_state")

    assert result == {"state_id": "snap-1"}
    assert session._conn.calls == [("save_state", None)]
