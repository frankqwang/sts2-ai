from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from google.protobuf import json_format

_python_root = Path(__file__).resolve().parents[1]
if str(_python_root) not in sys.path:
    sys.path.insert(0, str(_python_root))

import game_bridge
from game_bridge.generated import game_state_pb2 as pb
from game_bridge.session.base import SessionFactory
from game_bridge.session.game_session import GameSession, PipeProtoTransport, create_game_session
from game_bridge.session.pool import SessionPool
from game_bridge.spectate.controller import SpectatorController
from game_bridge.spectate.overlay import OverlayWriter
from game_bridge.spectate.policy import NullPolicy, ReplayPolicy
from game_bridge.transport.connection import PipeConnection, PipeConnectionConfig
from game_bridge.transport.proto_codec import ProtoCodec
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

    def _fake_game_session(**kwargs):
        mode = str(kwargs.get("mode"))
        calls.append((mode, kwargs))
        return f"{mode}-session"

    monkeypatch.setattr("game_bridge.session.create_game_session", _fake_game_session)

    combat_factory = SessionFactory(kind="combat", config=SessionConfig(port=2222, auto_launch=True))
    full_run_factory = SessionFactory(
        kind="full_run",
        config=SessionConfig(port=3333, transport="pipe_proto", base_url="http://localhost:9"),
    )

    assert combat_factory.create() == "combat-session"
    assert full_run_factory.create() == "full_run-session"
    assert calls[0][0] == "combat"
    assert calls[0][1]["port"] == 2222
    assert calls[1][0] == "full_run"
    assert calls[1][1]["port"] == 3333
    assert calls[1][1]["transport"] == "pipe_proto"


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


def test_game_bridge_source_has_no_legacy_protocol_symbols():
    runtime_root = _python_root / "game_bridge"
    forbidden = (
        "PipeRequestEnvelope",
        "PipeResponseEnvelope",
        "JsonCodec",
        "SingleplayerClient",
        "BinaryBackedFullRunClient",
        "PipeBackedFullRunClient",
        "ApiBackedFullRunClient",
        "/api/v2/full_run_env",
        "/api/v1/singleplayer",
    )
    for path in list(runtime_root.rglob("*.py")) + list((_python_root / "scripts").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, f"{path} still references {marker}"


def test_full_run_auto_launch_defaults_are_resolved():
    client = create_game_session(mode="full_run", transport="pipe_proto", backend="sim", port=17777, auto_launch=True)
    try:
        assert isinstance(client, GameSession)
        assert isinstance(client._transport, PipeProtoTransport)
        assert client._transport.repo_root is not None
        assert client._transport.host_path is not None
        assert client._transport._conn.cfg.auto_launch is True
        assert client._transport._conn.cfg.sim_launcher is not None
    finally:
        client.close()


def test_create_game_session_rejects_unknown_transport():
    with pytest.raises(ValueError, match="Unsupported GameSession transport"):
        create_game_session(mode="full_run", transport="pipe_json", backend="sim")


def test_combat_session_snapshot_methods_reuse_pipe_mixin():
    session = object.__new__(GameSession)
    calls: list[tuple[str, dict[str, object] | None]] = []
    session.mode = "combat"
    session._current_state = {"state_type": "stale"}

    def _fake_call(method: str, params: dict[str, object] | None = None):
        calls.append((method, params))
        if method == "save_search_state":
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
        "save_search_state",
        "load_state",
        "export_state",
        "import_state",
        "delete_state",
    ]


def test_combat_session_load_state_refreshes_current_state():
    session = object.__new__(GameSession)
    session._current_state = {"state_type": "stale", "legal_actions": []}

    def _fake_call(method: str, params: dict[str, object] | None = None):
        assert method == "load_state"
        assert params == {"state_id": "combat-state-2"}
        return {
            "state_type": "monster",
            "terminal": False,
            "legal_actions": [{"action": "play_card", "card_index": 0}],
        }

    session._call = _fake_call

    restored = session.load_state("combat-state-2")

    assert restored["state_type"] == "monster"
    assert session.current_state["state_type"] == "monster"
    assert session.legal_actions == [{"action": "play_card", "card_index": 0}]


def test_combat_session_call_allows_snapshot_rpcs():
    class _FakeTransport:
        transport_name = "pipe_proto"

        def __init__(self):
            self.calls: list[tuple[str, dict[str, object] | None]] = []

        @property
        def last_call_metrics(self):
            return {}

        def connect(self):
            return None

        def close(self):
            return None

        def call(self, method, params=None, *, timeout_s=None):
            self.calls.append((method, params))
            return {"state_id": "snap-1"}

    session = object.__new__(GameSession)
    session._transport = _FakeTransport()

    result = GameSession._call(session, "save_search_state")

    assert result == {"state_id": "snap-1"}
    assert session._transport.calls == [("save_search_state", None)]


def test_pipe_connection_auto_launch_uses_short_probe_timeout(monkeypatch: pytest.MonkeyPatch):
    connect_timeouts: list[float] = []
    launcher_calls: list[int] = []

    class _FakeTransport:
        connect_attempts = 0

        def __init__(self, _pipe_name: str):
            self._connected = False

        def connect(self, timeout_s: float = 10.0):
            connect_timeouts.append(timeout_s)
            type(self).connect_attempts += 1
            if type(self).connect_attempts == 1:
                raise ConnectionError("pipe not ready")
            self._connected = True

        def is_connected(self):
            return self._connected

        def close(self):
            self._connected = False

    monkeypatch.setattr("game_bridge.transport.connection.PipeTransport", _FakeTransport)
    monkeypatch.setattr(PipeConnection, "_handshake", lambda self: None)

    cfg = PipeConnectionConfig(
        port=19999,
        protocol="proto",
        connect_timeout_s=5.0,
        auto_launch=True,
        sim_launcher=lambda port: launcher_calls.append(port) or object(),
    )
    conn = PipeConnection(cfg)

    conn.connect()

    assert launcher_calls == [19999]
    assert connect_timeouts == [1.0, 15.0]
    assert conn.is_connected() is True


def test_proto_codec_combat_reset_preserves_optional_build_presence():
    codec = ProtoCodec()
    payload = codec.encode_request(
        "combat_reset",
        {
            "character_id": "IRONCLAD",
            "encounter_id": "CHOMPERS_NORMAL",
            "build": {
                "deck": [{"id": "STRIKE_IRONCLAD", "upgrade_level": 0}],
                "relics": [{"id": "BURNING_BLOOD"}],
                "potions": [{"id": "FIRE_POTION", "slot": 1}],
                "gold": 0,
            },
        },
    )

    req = pb.BridgeRequestEnvelope()
    req.ParseFromString(payload)

    assert req.method == pb.COMBAT_RESET
    assert req.combat_reset.build.HasField("gold") is True
    assert req.combat_reset.build.gold == 0
    assert req.combat_reset.build.HasField("current_hp") is False
    assert req.combat_reset.build.HasField("max_hp") is False
    assert req.combat_reset.build.HasField("max_energy") is False
    assert req.combat_reset.build.HasField("max_potion_slots") is False
    assert len(req.combat_reset.build.potions) == 1
    assert req.combat_reset.build.potions[0].id == "FIRE_POTION"
    assert req.combat_reset.build.potions[0].slot == 1


def test_bridge_envelope_round_trips_through_protobuf_json_mapping():
    codec = ProtoCodec()
    request = codec.build_request_message(
        "act",
        {"action": "play_card", "card_index": 1, "target_id": 2, "label": "Strike"},
    )
    request_json = json_format.MessageToJson(request)
    parsed_request = pb.BridgeRequestEnvelope()
    json_format.Parse(request_json, parsed_request)

    assert parsed_request.method == pb.ACT
    assert parsed_request.act.action.action == "play_card"
    assert parsed_request.act.action.card_index == 1
    assert parsed_request.act.action.target_id == 2

    response = pb.BridgeResponseEnvelope(
        method=pb.ACT,
        status=pb.OK,
        act=pb.BridgeActPayload(
            accepted=True,
            state=pb.GameState(state_type="monster", terminal=False),
            settlement_events=[
                pb.SettlementEvent(
                    type="damage_received",
                    sequence=0,
                    round_number=1,
                    turn_side="player",
                    actor_id="IRONCLAD",
                    actor_is_player=True,
                    target_id="CULTIST",
                    target_combat_id=1,
                    unblocked_damage=8,
                    total_damage=8,
                    source_card_id="BASH",
                    description="Rd 1 (Player turn): IRONCLAD dealt 8 damage to CULTIST.",
                )
            ],
        ),
    )
    response_json = json_format.MessageToJson(response)
    parsed_response = pb.BridgeResponseEnvelope()
    json_format.Parse(response_json, parsed_response)
    decoded = codec.decode_response_message(parsed_response)

    assert decoded["accepted"] is True
    assert decoded["state"]["state_type"] == "monster"
    assert decoded["settlement_events"][0]["type"] == "damage_received"
    assert decoded["settlement_events"][0]["unblocked_damage"] == 8
    assert decoded["info"]["settlement_events"][0]["source_card_id"] == "BASH"


def test_game_session_act_records_last_settlement_events():
    class _FakeTransport:
        transport_name = "pipe_proto"

        @property
        def last_call_metrics(self):
            return {}

        def close(self):
            return None

        def call(self, method, params=None, *, timeout_s=None):
            assert method == "act"
            return {
                "accepted": True,
                "state": {"state_type": "monster", "terminal": False},
                "settlement_events": [
                    {
                        "type": "power_received",
                        "target_id": "CULTIST",
                        "power_id": "VULNERABLE",
                        "amount_value": 2.0,
                    }
                ],
                "info": {"state_type": "monster"},
            }

    session = object.__new__(GameSession)
    session.mode = "full_run"
    session._transport = _FakeTransport()
    session._current_state = {}
    session._last_step_info = None
    session._last_settlement_events = []

    state = session.act({"action": "play_card", "card_index": 0, "target_id": 1})

    assert state["state_type"] == "monster"
    assert session.last_settlement_events[0]["power_id"] == "VULNERABLE"
    assert session.last_step_info["settlement_events"][0]["amount_value"] == 2.0
