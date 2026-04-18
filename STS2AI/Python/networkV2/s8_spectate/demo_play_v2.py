"""NetworkV2 observ mode / demo_play — 用 networkV2 UnifiedNet checkpoint 跑一把完整 run
并通过 Godot spectator 可视化。

和 V1 `demo_play.py` 的关系:
  - 共用 Godot 连接协议（FullRunClientLike via `create_full_run_client`）
  - 替换 V1 推理（combat_network + fullrun_policy）为 V2 UnifiedNet + CombatFeatureCompiler
  - 流程跟 `train_full_run_v2.run_full_episode` 一致，只是 client 从训练 sim 换成 Godot

用法（前提：Godot spectator 已在 port 15526 监听，或按 --base-url）:
  python -m networkV2.s8_spectate.demo_play_v2 \\
      --checkpoint ../Artifacts/checkpoints/co21/cotrainer_iter140.pt \\
      --base-url http://127.0.0.1:15526 \\
      --character-id IRONCLAD --greedy

或用 pipe transport:
  python -m networkV2.s8_spectate.demo_play_v2 \\
      --checkpoint ... --use-pipe --port 15527 --transport pipe-proto
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.distributions import Categorical

from networkV2.s3_state_tracker.combat_env_wrapper import CombatStateTracker
from networkV2.s4_compiler.feature_compiler import CombatFeatureCompiler
from networkV2.s5_net.network_config import from_preset
from networkV2.s5_net.unified_net import UnifiedNet
from networkV2.s1_schema.encounter_vocab import encounter_to_index
from networkV2.s1_schema.sim_catalog import GAME_CATALOG

from env.full_run_env import create_full_run_client

logger = logging.getLogger(__name__)

COMBAT_TYPES = {"monster", "elite", "boss", "hand_select", "card_select"}


class OverlayWriter:
    """写决策到 JSON 文件，Godot 通过 --mcp-decision-overlay-file 读取并叠加显示。"""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, data: dict) -> None:
        if self.path is None:
            return
        import json as _json
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                _json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"overlay write failed: {e}")


def _derive_encounter_id(state: dict) -> str:
    """跟 train_full_run_v2 / combat_cotrainer 一致的 encounter_id 推导规则。"""
    eid = str(state.get("encounter_id", "") or "").lower().strip()
    if eid:
        return eid
    battle = state.get("battle") or {}
    enemies = battle.get("enemies") or state.get("enemies") or []
    eids = sorted({
        str(e.get("monster_id") or e.get("enemy_id") or e.get("id") or "").upper()
        for e in enemies if isinstance(e, dict)
    })
    eids = [x for x in eids if x]
    return ",".join(eids).lower()


def _detect_room_type(state: dict) -> str:
    st = str(state.get("state_type", "")).lower()
    if "boss" in st:
        return "boss"
    if "elite" in st:
        return "elite"
    if st in ("monster", "hand_select", "card_select"):
        return "monster"
    return st


def play_one_episode(
    client,
    net: UnifiedNet,
    compiler: CombatFeatureCompiler,
    *,
    character_id: str = "IRONCLAD",
    seed: str | None = None,
    max_steps: int = 800,
    greedy: bool = False,
    step_delay: float = 0.0,
    combat_delay: float = 0.0,
    overlay: "OverlayWriter | None" = None,
) -> dict:
    state = client.reset(character_id=character_id, seed=seed)

    tracker = CombatStateTracker()
    tracker.on_run_start()
    in_combat = False
    device = next(net.parameters()).device
    step_count = 0
    combat_count = 0
    idle_polls = 0

    while step_count < max_steps:
        st_low = str(state.get("state_type", "")).lower()
        terminal = state.get("terminal", False)
        outcome = state.get("run_outcome")
        # 终局：game_over state 或 terminal flag 或 run_outcome 确定
        if st_low == "game_over" or terminal or outcome:
            if in_combat:
                tracker.on_combat_end(state)
            logger.info(f"Episode end: state_type={st_low} terminal={terminal} outcome={outcome}")
            break

        # 只取 is_enabled != False 的 legal actions
        legal = [a for a in state.get("legal_actions", []) or []
                 if isinstance(a, dict) and a.get("is_enabled") is not False]
        if not legal:
            # 过渡 state（loading/animation 等），poll 重试而非退出
            idle_polls += 1
            if idle_polls > 40:  # 40 × 0.25s = 10s 仍没 legal，才退出
                logger.warning(f"No legal actions after {idle_polls} polls; state_type={st_low}; exiting.")
                break
            time.sleep(0.25)
            try:
                state = client.get_state()
            except Exception as e:
                logger.warning(f"get_state failed during idle: {e}")
                time.sleep(0.5)
            continue
        idle_polls = 0

        st = str(state.get("state_type", "")).lower()
        is_combat = st in COMBAT_TYPES
        rt = _detect_room_type(state) if is_combat else st

        # combat boundary transitions
        if is_combat and not in_combat:
            eid = _derive_encounter_id(state)
            tracker.on_combat_start(state, eid, rt)
            in_combat = True
            combat_count += 1
            logger.info(f"Combat {combat_count} START: {eid} ({rt})")
        elif not is_combat and in_combat:
            tracker.on_combat_end(state)
            in_combat = False

        # 非战斗 RBM 更新
        if not is_combat and st in ("shop", "rest_site", "map", "event", "card_reward", "combat_rewards"):
            room_kind = {"rest_site": "rest", "combat_rewards": "card_reward"}.get(st, st)
            rbm = tracker.run_build_memory
            if not rbm.room_type_history or rbm.room_type_history[-1] != room_kind:
                rbm.register_room(room_kind)
                if st == "event":
                    ev = str(state.get("event_id", state.get("encounter_id", "")) or "")
                    rbm.register_event(ev)
            tracker.refresh_build_profile(state)

        cur_encounter_id = tracker.encounter_id if is_combat else ""
        banks = compiler.compile(
            state, legal,
            combat_memory=tracker.combat_memory if is_combat else None,
            turn_prefix=tracker.turn_prefix if is_combat else None,
            run_build_memory=tracker.run_build_memory,
            encounter_id=cur_encounter_id,
            room_type=rt if is_combat else "monster",
        )

        enc_idx_tensor = torch.tensor(
            [encounter_to_index(cur_encounter_id)], dtype=torch.long, device=device,
        )
        with torch.no_grad():
            out = net(banks=banks, encounter_idx=enc_idx_tensor)
        logits = out.logits[0, :len(legal)]
        mask = out.action_mask[0, :len(legal)]
        logits = torch.nan_to_num(logits.masked_fill(~mask, float("-inf")), nan=0.0)

        if greedy:
            idx = int(logits.argmax().item())
        else:
            dist = Categorical(logits=logits)
            idx = int(dist.sample().item())
        chosen = legal[idx]

        action_label = str(chosen.get("label") or chosen.get("action", "?"))[:60]
        card_id = chosen.get("card_id", "") if chosen.get("action") == "play_card" else ""
        # HP 多路径 fallback (Godot state 可能在 battle.player / state.player 等)
        player = (state.get("player") or {})
        battle = state.get("battle") or {}
        hp = (player.get("hp") or battle.get("player_hp")
              or (battle.get("player") or {}).get("hp") or "?")
        mhp = (player.get("max_hp") or battle.get("player_max_hp")
               or (battle.get("player") or {}).get("max_hp") or "?")
        # 发 decision overlay 给 Godot
        if overlay is not None:
            overlay.publish({
                "msg_type": "decision",
                "step": step_count,
                "state_type": st,
                "floor": (state.get("run") or {}).get("floor", 0),
                "hp": hp, "max_hp": mhp,
                "action": chosen.get("action", "?"),
                "action_label": action_label,
                "card_id": card_id,
                "chosen_idx": idx,
                "n_legal": len(legal),
                "room_type": rt if is_combat else st,
                "turn": int(tracker.combat_memory.turn_index) if is_combat else 0,
                "encounter_id": tracker.encounter_id or "",
                "reasoning_zh": (
                    f"战斗 {combat_count} 回合 {tracker.combat_memory.turn_index}: "
                    f"HP {hp}/{mhp}  → {action_label}"
                    + (f" ({card_id})" if card_id else "")
                ) if is_combat else (
                    f"[非战斗] floor {(state.get('run') or {}).get('floor', 0)} "
                    f"{st}  → {action_label}"
                ),
            })
        if is_combat:
            battle = state.get("battle") or {}
            block = battle.get("block", player.get("block", 0))
            energy = battle.get("energy", player.get("energy", 0))
            print(f"[combat {combat_count} step {step_count}] "
                  f"{tracker.encounter_id} T{tracker.combat_memory.turn_index}  "
                  f"HP {hp}/{mhp} blk {block} E{energy}  ->  {action_label}"
                  + (f"  ({card_id})" if card_id else ""))
        else:
            floor = (state.get("run") or {}).get("floor", 0)
            print(f"[noncombat step {step_count}] {st} floor {floor}  HP {hp}/{mhp}  ->  {action_label}")

        # 发送 action
        next_state = client.act(chosen)

        if is_combat:
            tracker.on_step(next_state, chosen, prev_state=state)

        state = next_state
        step_count += 1

        delay = combat_delay if is_combat else step_delay
        if delay > 0:
            time.sleep(delay)

    final = {
        "steps": step_count,
        "combats": combat_count,
        "outcome": str(state.get("run_outcome") or "unknown"),
        "floor": (state.get("run") or {}).get("floor", 0),
        "final_hp": (state.get("player") or {}).get("hp", 0),
    }
    print()
    print("=" * 72)
    print(f"  END: outcome={final['outcome']}  floor={final['floor']}  "
          f"HP={final['final_hp']}  combats={final['combats']}  steps={final['steps']}")
    print("=" * 72)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="networkV2 UnifiedNet checkpoint (.pt)")
    ap.add_argument("--preset", type=str, default="slim",
                    choices=["slim", "full", "tiny"])
    ap.add_argument("--base-url", type=str, default="http://127.0.0.1:15526",
                    help="Godot spectator HTTP base url (http transport)")
    ap.add_argument("--use-pipe", action="store_true",
                    help="Use pipe transport instead of HTTP")
    ap.add_argument("--port", type=int, default=15527,
                    help="Pipe port (use-pipe 时用)")
    ap.add_argument("--transport", type=str, default=None,
                    choices=[None, "http", "pipe", "pipe-proto"],
                    help="pipe 协议(use-pipe 时)。pipe-binary 已废弃,诊断用 pipe(json),训练用 pipe-proto")
    ap.add_argument("--character-id", type=str, default="IRONCLAD")
    ap.add_argument("--seed", type=str, default=None)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--step-delay", type=float, default=0.0,
                    help="非战斗每 step 间隔秒")
    ap.add_argument("--combat-delay", type=float, default=0.0,
                    help="战斗内每 step 间隔秒（观战时建议 0.3-0.5 让人看清）")
    ap.add_argument("--greedy", action="store_true",
                    help="argmax action 而非采样（观战建议开）")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--decision-overlay-file", type=str, default=None,
                    help="写 AI 决策到 JSON 文件 (Godot --mcp-decision-overlay-file 读取)")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    # 1) 建网络
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = from_preset(args.preset)
    net = UnifiedNet(config=cfg).to(device)
    params = sum(p.numel() for p in net.parameters())
    logger.info(f"UnifiedNet preset={args.preset}  {params:,} params ({params/1e6:.1f}M)")

    # 2) Load checkpoint（兼容 partial load）
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "net" in state:
        state = state["net"]
    try:
        net.load_state_dict(state)
        logger.info(f"Loaded checkpoint (full): {args.checkpoint}")
    except Exception as e:
        logger.warning(f"Full load failed ({type(e).__name__}); trying compatible...")
        report = net.load_compatible_params(state, strict_shapes=True)
        logger.info(f"Partial load: loaded={report['loaded']} "
                    f"skipped_shape={report['skipped_shape']} missing={report['missing']}")
    net.eval()

    # 3) 连 Godot
    client = create_full_run_client(
        base_url=args.base_url,
        port=args.port,
        use_pipe=args.use_pipe,
        transport=args.transport,
    )
    logger.info(f"Connected spectator via {'pipe' if args.use_pipe else 'http'} "
                f"({args.base_url if not args.use_pipe else f'port {args.port}'})")

    # 4) 附加 game catalog（encounter_vocab / card 存在性校验需要）
    try:
        GAME_CATALOG.attach_sim(client)
        logger.info("Attached GAME_CATALOG to spectator client")
    except Exception as e:
        logger.warning(f"GAME_CATALOG attach failed (some features may use sqlite fallback): {e}")

    compiler = CombatFeatureCompiler()

    # overlay
    overlay = OverlayWriter(Path(args.decision_overlay_file)) if args.decision_overlay_file else None
    if overlay:
        logger.info(f"Overlay → {args.decision_overlay_file}")

    # 5) 跑一把
    info = play_one_episode(
        client, net, compiler,
        character_id=args.character_id,
        seed=args.seed,
        max_steps=args.max_steps,
        greedy=args.greedy,
        step_delay=args.step_delay,
        combat_delay=args.combat_delay,
        overlay=overlay,
    )
    logger.info(f"Episode done: {info}")


if __name__ == "__main__":
    main()
