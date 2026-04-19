"""CUDA graph capture 定位工具:逐层 capture 定位不 capture-safe 的 op。

用法:
    cd STS2AI/Python && python -m networkV2.s5_net.graph_debug

输出类似:
    stage 0 (project_static only):     OK
    stage 1 (+ _empty_bt):              OK
    stage 2 (+ build_encoder):          FAIL: cudaErrorStreamCaptureInvalidated
                                        → 定位到 build_encoder 内部
"""
from __future__ import annotations

import sys
import torch


def _prepare():
    """Build net + static buffers + sample banks."""
    from networkV2.s5_net.network_config import from_preset
    from networkV2.s5_net.unified_net import UnifiedNet
    from networkV2.s4_featurization.decision_featurizer import DecisionFeaturizer
    from networkV2.s5_net.tokenizer import alloc_static_bank_buffers
    from networkV2.s5_net.bank_max_spec import DEFAULT_MAX_SPEC

    cfg = from_preset("slim")
    net = UnifiedNet(config=cfg).cuda().eval()

    obs = {
        "state_type": "monster",
        "player": {
            "hp": 60, "max_hp": 80, "energy": 3, "max_energy": 3, "block": 5,
            "deck": [{"id": "STRIKE_IRONCLAD", "type": "ATTACK", "rarity": "common",
                      "cost": 1}] * 30,
            "relics": [{"id": "BURNING_BLOOD"}], "potions": [],
        },
        "hand": [{"id": "STRIKE_IRONCLAD", "type": "ATTACK", "cost": 1,
                  "can_play": True, "damage": 6}] * 10,
        "battle": {
            "enemies": [{"id": "JAW_WORM", "hp": 42, "max_hp": 42, "block": 0,
                         "intent": "attack", "powers": []}] * 4,
        },
    }
    legal = [{"action": "play_card", "hand_index": i, "target_id": j}
             for i in range(10) for j in range(4)] + [{"action": "end_turn"}]
    featurizer = DecisionFeaturizer()
    banks = featurizer.featurize(obs, legal, encounter_id="jaw_worm_easy",
                             room_type="monster")

    bank_names = [b.bank_name for b in banks.all_banks()
                  if not b.is_empty and hasattr(DEFAULT_MAX_SPEC, b.bank_name)]
    host, gpu = alloc_static_bank_buffers(
        bank_names=bank_names,
        max_spec=DEFAULT_MAX_SPEC,
        device=torch.device("cuda"),
        max_numeric_dim=net.tokenizer.max_numeric_dim,
    )
    net.tokenizer.fill_static_buffers(banks, host, gpu)
    enc_idx = torch.zeros(1, dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    return net, gpu, banks, enc_idx


def _try_capture(name, callable_):
    """尝试 capture callable_()。成功打 OK,失败打错误类型。"""
    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    _ = callable_()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  {name:60s} WARMUP FAIL: {type(e).__name__}: {str(e)[:80]}")
        return False

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            with torch.no_grad():
                _ = callable_()
        print(f"  {name:60s} OK")
        return True
    except Exception as e:
        print(f"  {name:60s} FAIL: {type(e).__name__}: {str(e)[:80]}")
        # 清理
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False


def main():
    print("=== CUDA graph capture layer-by-layer ===")
    net, gpu, banks, enc_idx = _prepare()
    tok = net.tokenizer

    # Stage 0: 仅 project_static
    _try_capture(
        "stage 0: tokenizer.project_static(gpu)",
        lambda: tok.project_static(gpu),
    )

    # Stage 1: project + _empty_bt
    def _stage1():
        bt = tok.project_static(gpu)
        device = torch.device("cuda")
        _ = net._empty_bt(device, 1)
        return bt
    _try_capture("stage 1: + _empty_bt", _stage1)

    # Stage 2: + build_encoder
    def _stage2():
        bt = tok.project_static(gpu)
        device = torch.device("cuda")
        _e = net._empty_bt(device, 1)
        build_bt = bt.get("build", _e)
        inventory_bt = bt.get("inventory")
        return net.build_encoder(build_bt, inventory_bt)
    _try_capture("stage 2: + build_encoder", _stage2)

    # Stage 3: + masked_mean + shared_world_proj
    def _stage3():
        bt = tok.project_static(gpu)
        device = torch.device("cuda")
        _e = net._empty_bt(device, 1)
        objective_bt = bt.get("objective", _e)
        forecast_bt = bt.get("forecast", _e)
        obj_pool = net._masked_mean(objective_bt)
        fcast_pool = net._masked_mean(forecast_bt)
        world_ctx = net.shared_world_proj(torch.cat([obj_pool, fcast_pool], dim=-1))
        return world_ctx
    _try_capture("stage 3: + masked_mean + shared_world_proj", _stage3)

    # Stage 4: + encoders (board/mech/mod/power/prefix/cm)
    def _stage4():
        bt = tok.project_static(gpu)
        device = torch.device("cuda")
        _e = net._empty_bt(device, 1)
        # combat-only banks
        board_bt = bt.get("board", _e)
        mech_bt = bt.get("mechanism", _e)
        mod_bt = bt.get("modifier", _e)
        power_bt = bt.get("power", _e)
        prefix_bt = bt.get("turn_prefix", _e)
        cm_bt = bt.get("combat_memory", _e)
        return (
            net.board_encoder(board_bt),
            net.mechanism_encoder(mech_bt),
            net.modifier_encoder(mod_bt),
            net.power_encoder(power_bt),
            net.prefix_encoder(prefix_bt),
            net.combat_memory_encoder(cm_bt),
        )
    _try_capture("stage 4: + combat encoders (6)", _stage4)

    # Stage 5: + action_contextualizer
    def _stage5():
        bt = tok.project_static(gpu)
        device = torch.device("cuda")
        _e = net._empty_bt(device, 1)
        build_bt = bt.get("build", _e)
        inventory_bt = bt.get("inventory")
        build_slots = net.build_encoder(build_bt, inventory_bt)
        action_bt = bt.get("action", _e)
        board_enc = net.board_encoder(bt.get("board", _e))
        mech_enc = net.mechanism_encoder(bt.get("mechanism", _e))
        mod_enc = net.modifier_encoder(bt.get("modifier", _e))
        power_enc = net.power_encoder(bt.get("power", _e))
        prefix_enc = net.prefix_encoder(bt.get("turn_prefix", _e))
        cm_enc = net.combat_memory_encoder(bt.get("combat_memory", _e))
        return net.action_contextualizer(
            action_bt, board_enc, mod_enc, mech_enc, power_enc, prefix_enc, cm_enc, build_slots,
        )
    _try_capture("stage 5: + action_contextualizer", _stage5)

    # Stage 6: + decision_core
    def _stage6():
        bt = tok.project_static(gpu)
        device = torch.device("cuda")
        _e = net._empty_bt(device, 1)
        build_bt = bt.get("build", _e)
        build_slots = net.build_encoder(build_bt, bt.get("inventory"))
        action_bt = bt.get("action", _e)
        action_hyp = net.action_contextualizer(
            action_bt,
            net.board_encoder(bt.get("board", _e)),
            net.modifier_encoder(bt.get("modifier", _e)),
            net.mechanism_encoder(bt.get("mechanism", _e)),
            net.power_encoder(bt.get("power", _e)),
            net.prefix_encoder(bt.get("turn_prefix", _e)),
            net.combat_memory_encoder(bt.get("combat_memory", _e)),
            build_slots,
        )
        return net.decision_core(action_hyp)
    _try_capture("stage 6: + decision_core", _stage6)

    # Stage 7: + encounter_conditioning
    def _stage7():
        bt = tok.project_static(gpu)
        device = torch.device("cuda")
        _e = net._empty_bt(device, 1)
        build_slots = net.build_encoder(bt.get("build", _e), bt.get("inventory"))
        action_bt = bt.get("action", _e)
        action_hyp = net.action_contextualizer(
            action_bt,
            net.board_encoder(bt.get("board", _e)),
            net.modifier_encoder(bt.get("modifier", _e)),
            net.mechanism_encoder(bt.get("mechanism", _e)),
            net.power_encoder(bt.get("power", _e)),
            net.prefix_encoder(bt.get("turn_prefix", _e)),
            net.combat_memory_encoder(bt.get("combat_memory", _e)),
            build_slots,
        )
        decision_repr, action_refined = net.decision_core(action_hyp)
        return net._apply_encounter_conditioning(decision_repr, enc_idx)
    _try_capture("stage 7: + encounter_conditioning", _stage7)

    # Stage 8: full forward_from_static
    def _stage8():
        return net.forward_from_static(gpu, enc_idx, decision_domain="combat")
    _try_capture("stage 8: full forward_from_static", _stage8)


if __name__ == "__main__":
    main()
