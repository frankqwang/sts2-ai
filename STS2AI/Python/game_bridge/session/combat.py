"""CombatSession — combat 训练的唯一高层 API,proto 直连。

## 为什么需要这一层

历史上 combat 训练直接用 `env.combat_training_env.PipeBackedCombatTrainingClient`
(JSON + 手写 schema 解析)。每个训练/评测脚本都重复:
  - `_normalize_hand_card` (PascalCase ↔ snake_case fallback)
  - `_TARGET_TYPE_NAMES` (手写 enum map, 2026-04-18 发现错位一位的 bug)
  - `build_combat_legal_actions` (Python 自己从 hand 推断 legal actions,
    和 sim 端的 step 验证逻辑可能不一致)

**2026-04-18 重构**:CombatSession 改走 proto wire,sim 端(Program.cs +
ProtoStateBuilder.BuildCombatStateResponse)直接 populate `GameState.legal_actions`,
Python 侧不再自己推断,彻底消除 schema drift。

## 权威字段优先级

按 `docs/design/SCHEMA_CONVENTION.md`:
  1. Sim runtime API (sim 发的 `RequiresTarget` / `ValidTargetIds`)  ← 最权威
  2. sqlite (fallback)
  3. 手写常量  ← 禁止

本 session **完全信任 sim 字段**。不再依赖 `_TARGET_TYPE_NAMES` enum map。

## Proto wire 细节

- 底层: `PipeConnection + ProtoCodec`(transport 层,不允许造轮子)
- pipe 名: `sts2_mcts_proto_{port}`(sim 启动 `--protocol proto`)
- opcode: `CombatReset=0x11 / CombatStep=0x12 / CombatState=0x13`
- 请求: `CombatResetRequest / CombatStepRequest` protobuf 消息
- 响应: `GameState` protobuf (含 `legal_actions` 权威字段)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from game_bridge.generated import game_state_pb2 as pb
from game_bridge.session.build_spec import BuildSpecPy, CardSpecPy, RelicSpecPy
from game_bridge.transport import (
    PipeConnection,
    PipeConnectionConfig,
    ProtoCodec,
    SimulatorApiError,
)
from game_bridge.session.state_semantics import is_failure_outcome, is_victory_outcome, normalize_run_outcome

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CombatSession - 唯一对外 combat API
# ---------------------------------------------------------------------------

class CombatSession:
    """Combat 训练的统一 session (proto wire)。

    Usage:
        session = CombatSession(port=18000, auto_launch=True)
        session.reset(character="IRONCLAD", encounter="CHOMPERS_NORMAL",
                      build=BuildSpecPy(deck=[...], relics=[...], current_hp=60, max_hp=80))
        while not session.terminal:
            state = session.current_state
            legal = state["legal_actions"]  # sim 权威 list[dict]
            action = pick_action(legal)
            state = session.step(action)
        session.close()

    **权威 legal_actions**:sim 的 ProtoStateBuilder.PopulateCombatLegalActions 直接
    填 GameState.legal_actions。Python 不再推断,消除 JSON path 历史的 enum 错位 bug。

    线程模型:一个 session 一 thread。多 worker 各建一个 session。
    """

    def __init__(
        self,
        *,
        port: int,
        auto_launch: bool = True,
        connect_timeout_s: float = 15.0,
        repo_root: str | Path | None = None,
        host_path: str | Path | None = None,
        dll_path: str | Path | None = None,
    ):
        # 延迟 import 避免循环 import
        from game_bridge.sim.launcher import (
            DEFAULT_HOST_PATH, DEFAULT_REPO_ROOT, start_headless_sim, stop_process,
        )
        self._repo_root = repo_root or DEFAULT_REPO_ROOT
        self._host_path = host_path or dll_path or DEFAULT_HOST_PATH
        self._auto_launch = auto_launch
        self._connect_timeout_s = float(connect_timeout_s)
        self._port = int(port)
        self._stop_process = stop_process
        self._owned_proc: Any | None = None

        def _launcher(launch_port: int):
            proc = start_headless_sim(
                port=launch_port,
                repo_root=self._repo_root,
                host_path=self._host_path,
                connect_timeout_s=max(20.0, self._connect_timeout_s),
                protocol="proto",
            )
            self._owned_proc = proc
            return proc

        cfg = PipeConnectionConfig(
            port=self._port,
            protocol="proto",
            connect_timeout_s=self._connect_timeout_s,
            auto_launch=auto_launch,
            sim_launcher=_launcher if auto_launch else None,
            sim_stopper=stop_process if auto_launch else None,
            codec=ProtoCodec(),
        )
        self._conn = PipeConnection(cfg)
        self._current_raw: dict[str, Any] = {}
        self._current_build: BuildSpecPy | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        # PipeConnection.close 已经 stop sim;这里只兜底 owned_proc
        if self._owned_proc is not None:
            try:
                self._stop_process(self._owned_proc)
            except Exception:
                pass
            self._owned_proc = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # API 兼容 PipeBackedCombatTrainingClient:reset / step / combat_catalog /
    # get_state / close。训练代码无需区分 JSON 与 proto。
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        encounter_id: str | None = None,
        character_id: str | None = None,
        character: str | None = None,
        encounter: str | None = None,
        build: BuildSpecPy | dict | None = None,
        ascension: int | None = None,
        ascension_level: int | None = None,
        seed: str | None = None,
    ) -> dict[str, Any]:
        """开始新战斗。返回 current_state (含 sim 权威 legal_actions)。

        参数 alias:
          - character / character_id 都可传
          - encounter / encounter_id 都可传
          - ascension / ascension_level 都可传
        """
        if isinstance(build, dict):
            build = BuildSpecPy.from_dict(build)
        self._current_build = build
        if not self._conn.is_connected():
            self._conn.connect()
        params: dict[str, Any] = {
            "character_id": str(character_id or character or "IRONCLAD"),
            "encounter_id": str(encounter_id or encounter or ""),
            "ascension_level": int(ascension_level or ascension or 0),
        }
        if seed:
            params["seed"] = str(seed)
        if build is not None:
            params["build"] = build.to_sim_dict()
        raw = self._conn.safe_call("combat_reset", params)
        self._current_raw = _unwrap_state(raw)
        return self.current_state

    def step(self, action: pb.LegalAction | dict) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """执行一个 action。返回 (state, reward, done, info) 元组 (Gym-style)。

        和旧 PipeBackedCombatTrainingClient.step() 兼容的返回格式,combat_cotrainer
        零改动迁过来。action 可以是 proto LegalAction 或 dict。
        """
        if isinstance(action, pb.LegalAction):
            action_dict = {
                "action": action.action,
                "index": action.index,
                "card_index": action.card_index,
                "card_id": action.card_id,
                "label": action.label,
            }
            if action.target_id > 0:
                action_dict["target_id"] = action.target_id
            if action.col != 0:
                action_dict["col"] = action.col
            if action.row != 0:
                action_dict["row"] = action.row
            if action.slot != 0:
                action_dict["slot"] = action.slot
        else:
            action_dict = dict(action)

        if not self._conn.is_connected():
            self._conn.connect()
        result = self._conn.safe_call("combat_step", action_dict)
        if isinstance(result, dict) and not bool(result.get("accepted", True)):
            error = SimulatorApiError(str(result.get("error") or "Combat step rejected"))
            if isinstance(result.get("state"), dict):
                self._current_raw = result["state"]
                setattr(error, "latest_state", self.current_state)
            raise error
        state_dict = result.get("state") if isinstance(result, dict) else None
        self._current_raw = state_dict if isinstance(state_dict, dict) else _unwrap_state(result)
        state = self.current_state
        done = bool(state.get("terminal"))
        reward = 0.0
        outcome = normalize_run_outcome(state.get("run_outcome"))
        if done:
            if is_victory_outcome(outcome):
                reward = 1.0
            elif is_failure_outcome(outcome):
                reward = -1.0
        info = {
            "accepted": True,
            "state_type": state.get("state_type"),
            "run_outcome": outcome or None,
        }
        return state, reward, done, info

    def get_state(self) -> dict[str, Any]:
        """拿当前 state (不走 step)。沿用旧 PipeBackedCombatTrainingClient API。"""
        if not self._conn.is_connected():
            self._conn.connect()
        raw = self._conn.safe_call("combat_state", {})
        self._current_raw = _unwrap_state(raw)
        return self.current_state

    def combat_catalog(self) -> dict[str, Any]:
        """目前 proto wire 不支持 combat_catalog opcode,返回空让调用方 fallback sqlite。

        sim 侧后续会加 proto opcode(catalog 是 static 数据,不是 hot path)。
        combat_cotrainer 的 room_type lookup 已有 sqlite fallback (GAME_CATALOG)。
        """
        return {"encounters": []}

    def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """兼容 GameCatalog.attach_sim 的接口。

        proto wire 支持的 method (combat_reset/combat_step/combat_state) 直接转发;
        其他 method (game_catalog/combat_catalog 等 static 数据)目前 proto 不支持,
        raise NotImplementedError,上层 (`GameCatalog`) 自动 fallback 到 sqlite。
        """
        if method in {"combat_reset", "combat_step", "combat_state"}:
            if not self._conn.is_connected():
                self._conn.connect()
            return self._conn.safe_call(method, params)
        raise NotImplementedError(
            f"CombatSession (proto wire) does not support method {method!r}. "
            "Static catalog methods currently live only on JSON/gRPC paths — "
            "upper layers should fallback to sqlite."
        )

    # ------------------------------------------------------------------
    # State accessor
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> dict[str, Any]:
        """返回 current state dict。legal_actions 已经是 sim 权威字段。"""
        return dict(self._current_raw)

    @property
    def terminal(self) -> bool:
        return bool(self._current_raw.get("terminal"))

    @property
    def run_outcome(self) -> str:
        return str(normalize_run_outcome(self._current_raw.get("run_outcome")) or "")

    @property
    def legal_actions(self) -> list[dict[str, Any]]:
        la = self._current_raw.get("legal_actions")
        return list(la) if isinstance(la, list) else []


def _unwrap_state(result: Any) -> dict[str, Any]:
    """ProtoCodec 直接返回 GameState dict;某些 opcode 返回 envelope。保一致。"""
    if not isinstance(result, dict):
        return {}
    if "state" in result and isinstance(result["state"], dict):
        return result["state"]
    return result


__all__ = [
    "CombatSession",
    "BuildSpecPy",
    "CardSpecPy",
    "RelicSpecPy",
]
