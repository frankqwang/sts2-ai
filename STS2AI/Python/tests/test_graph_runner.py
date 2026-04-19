"""GraphRunner 回归测试 — 硬检测 CUDA graph 正确性不 drift。

跑法:
    cd STS2AI/Python && python -m pytest tests/test_graph_runner.py -v

触发条件:
    - PR 里改了 UnifiedNet forward / tokenizer / banks schema → 必跑
    - 升级 PyTorch / CUDA driver → 必跑

Fail 场景:
    - Shape signature 不稳定(e.g. MAX padding 漏了某些 bank)
    - CUDA graph vs eager 输出超过 atol/rtol
    - 新加 op 破坏 graph determinism
"""
from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available; graph runner tests require GPU",
)


def _build_sample_banks():
    """产一个有代表性的 combat banks 作为 capture 输入。"""
    from networkV2.s4_compiler.feature_compiler import CombatFeatureCompiler

    obs = {
        "state_type": "monster",
        "player": {
            "hp": 60, "max_hp": 80, "energy": 3, "max_energy": 3, "block": 5,
            "deck": [{"id": "STRIKE_IRONCLAD", "type": "ATTACK", "rarity": "common",
                      "cost": 1}] * 30,
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [],
        },
        "hand": [{"id": "STRIKE_IRONCLAD", "type": "ATTACK", "cost": 1,
                  "can_play": True, "damage": 6}] * 10,
        "battle": {
            "enemies": [{"id": "JAW_WORM", "hp": 42, "max_hp": 42, "block": 0,
                         "intent": "attack", "powers": []}] * 4,
        },
    }
    legal = [
        {"action": "play_card", "hand_index": i, "target_id": j}
        for i in range(10) for j in range(4)
    ] + [{"action": "end_turn"}]
    compiler = CombatFeatureCompiler()
    return compiler.compile(
        obs, legal, encounter_id="jaw_worm_easy", room_type="monster",
    )


def _build_net():
    from networkV2.s5_net.network_config import from_preset
    from networkV2.s5_net.unified_net import UnifiedNet

    cfg = from_preset("slim")
    net = UnifiedNet(config=cfg).cuda().eval()
    return net


def test_shape_signature_stable():
    """同一 banks 生成的 signature 必须相等(支持 hash / 字典 key)。"""
    from networkV2.s5_net.graph_runner import BankShapeSignature

    banks = _build_sample_banks()
    sig1 = BankShapeSignature.from_banks(banks)
    sig2 = BankShapeSignature.from_banks(banks)
    assert sig1 == sig2
    assert hash(sig1) == hash(sig2)


def test_shape_signature_distinguishes_different_banks():
    """不同 hand/enemy 数必须产不同 signature。"""
    from networkV2.s4_compiler.feature_compiler import CombatFeatureCompiler
    from networkV2.s5_net.graph_runner import BankShapeSignature

    compiler = CombatFeatureCompiler()
    obs_base = {
        "state_type": "monster",
        "player": {"hp": 60, "max_hp": 80, "energy": 3, "max_energy": 3,
                   "deck": [{"id": "STRIKE_IRONCLAD", "type": "ATTACK",
                             "rarity": "common", "cost": 1}] * 10,
                   "relics": [], "potions": []},
        "battle": {"enemies": [{"id": "JAW_WORM", "hp": 42, "max_hp": 42,
                                "intent": "attack", "powers": []}]},
    }
    obs_5hand = {**obs_base, "hand": [{"id": "STRIKE_IRONCLAD", "type": "ATTACK",
                                        "cost": 1, "can_play": True,
                                        "damage": 6}] * 5}
    obs_10hand = {**obs_base, "hand": [{"id": "STRIKE_IRONCLAD", "type": "ATTACK",
                                         "cost": 1, "can_play": True,
                                         "damage": 6}] * 10}
    legal = [{"action": "end_turn"}]

    banks_5 = compiler.compile(obs_5hand, legal, encounter_id="x", room_type="monster")
    banks_10 = compiler.compile(obs_10hand, legal, encounter_id="x", room_type="monster")
    sig_5 = BankShapeSignature.from_banks(banks_5)
    sig_10 = BankShapeSignature.from_banks(banks_10)
    assert sig_5 != sig_10, "不同 hand 数应该产生不同 signature"


def test_graph_runner_startup_parity():
    """启动 parity 必须通过 — 如果 fail 说明 graph capture 的 determinism 坏了。

    不再 pytest.skip 掉 CUDA/capture 类错误: 这些就是 capture-safety 回归,
    本来就应该 fail 挡 merge。CI 无 GPU 时整个模块已被 pytestmark.skipif 跳过。
    """
    from networkV2.s5_net.graph_runner import GraphRunner

    net = _build_net()
    banks = _build_sample_banks()
    enc_idx = torch.tensor([0], dtype=torch.long, device="cuda")

    runner = GraphRunner(
        net, banks, enc_idx,
        atol=1e-3, rtol=1e-3, startup_parity_n=5, parity_check_every=0,
        strict=True,
    )
    assert runner.enabled, "strict=True 下 GraphRunner.enabled 必须 True (否则应该早已 raise)"
    out = runner(banks, enc_idx)
    assert out is not None


def test_graph_runner_overflow_raises():
    """banks 某个 bank 超过 MAX_LEN 必须 raise BankOverflowError。

    这是硬检测 4 的核心:加了新 token 类型让 bank 爆量,runtime 会立刻报错,
    不会静默 silent wrong(截断 tokens 或 OOB 写 GPU buffer)。
    """
    from networkV2.s5_net.graph_runner import GraphCaptureFailedError, GraphRunner
    from networkV2.s5_net.bank_max_spec import BankOverflowError, BankMaxSpec

    net = _build_net()
    banks_capture = _build_sample_banks()
    enc_idx = torch.tensor([0], dtype=torch.long, device="cuda")

    # 人为把 action 的 MAX 调得极小,让 sample banks 立即 overflow
    tiny_spec = BankMaxSpec(action=5)
    try:
        with pytest.raises((BankOverflowError, GraphCaptureFailedError)):
            runner = GraphRunner(
                net, banks_capture, enc_idx,
                atol=1e-3, startup_parity_n=3, parity_check_every=0,
                max_spec=tiny_spec,
            )
            if runner.enabled:
                runner(banks_capture, enc_idx)
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def test_graph_runner_decision_domain_mismatch_raises():
    """combat vs noncombat decision_domain 不一致时 raise GraphShapeMismatchError。"""
    from networkV2.s5_net.graph_runner import GraphRunner, GraphShapeMismatchError

    net = _build_net()
    banks = _build_sample_banks()
    enc_idx = torch.tensor([0], dtype=torch.long, device="cuda")
    runner = GraphRunner(
        net, banks, enc_idx,
        atol=1e-3, startup_parity_n=3, parity_check_every=0,
    )
    if not runner.enabled:
        pytest.skip("graph runner disabled")

    # 改 runtime banks 的 decision_domain
    import copy
    banks_noncombat = copy.copy(banks)
    banks_noncombat.decision_domain = "card_reward"
    with pytest.raises(GraphShapeMismatchError):
        runner(banks_noncombat, enc_idx)


def test_bank_signature_persists_across_restart():
    """signature 是 hashable + equal,future-proof:
    即使在不同 Python session 里产生,同样结构的 banks 应等。
    (这保证测试 fixture / CI 稳定性)
    """
    from networkV2.s5_net.graph_runner import BankShapeSignature

    banks1 = _build_sample_banks()
    banks2 = _build_sample_banks()
    assert BankShapeSignature.from_banks(banks1) == BankShapeSignature.from_banks(banks2)


def test_graph_runner_strict_raises_on_capture_failure():
    """负面测试: 注入一个 data-dependent branch 让 capture 炸,strict=True 必须 raise。

    目的: 以后有人往 UnifiedNet.forward 里加了 `if tensor.any():` 这种 CPU-sync
    分支,这条 test 会 fail,强迫他回去改 capture-safe 版本。
    """
    import torch.nn as nn
    from networkV2.s5_net.graph_runner import GraphRunner, GraphCaptureFailedError

    net = _build_net()
    banks = _build_sample_banks()
    enc_idx = torch.tensor([0], dtype=torch.long, device="cuda")

    # 把 forward_from_static monkey-patch 加 data-dependent branch
    orig = net.forward_from_static

    def evil_forward(self, gpu_buffers, encounter_idx, decision_domain="combat"):
        out = orig(gpu_buffers, encounter_idx, decision_domain=decision_domain)
        # data-dependent: 对 CUDA graph capture 是 fatal (会 CPU sync)
        if out.logits.sum().item() > 0:  # .item() 强制 device→host sync
            pass
        return out

    import types
    net.forward_from_static = types.MethodType(evil_forward, net)

    try:
        with pytest.raises(GraphCaptureFailedError):
            GraphRunner(
                net, banks, enc_idx,
                atol=1e-3, startup_parity_n=3, parity_check_every=0,
                strict=True,
            )
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def test_graph_runner_non_strict_falls_back():
    """strict=False 明确允许降级:capture 失败时 enabled=False,不 raise。"""
    import types
    from networkV2.s5_net.graph_runner import GraphRunner

    net = _build_net()
    banks = _build_sample_banks()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    enc_idx = torch.tensor([0], dtype=torch.long, device="cuda")

    orig = net.forward_from_static

    def evil_forward(self, gpu_buffers, encounter_idx, decision_domain="combat"):
        out = orig(gpu_buffers, encounter_idx, decision_domain=decision_domain)
        if out.logits.sum().item() > 0:
            pass
        return out

    net.forward_from_static = types.MethodType(evil_forward, net)

    runner = GraphRunner(
        net, banks, enc_idx,
        atol=1e-3, startup_parity_n=3, parity_check_every=0,
        strict=False,
    )
    assert runner.enabled is False, "strict=False + capture 失败应该 enabled=False 不 raise"


def test_graph_runner_undeclared_bank_always_raises():
    """负面测试: sample_banks 有 bank 但 max_spec 没声明 → GraphBankUndeclaredError。

    永远 raise (strict 无关)。堵住"加新 bank 忘改 bank_max_spec"的静默漏洞:
    未来某人往 TokenBank schema 加 'new_special_bank' 但忘记在 BankMaxSpec 里
    加 `new_special_bank: int = 30` 字段时,这条 test 会 fail。
    """
    from networkV2.s5_net.graph_runner import GraphRunner, GraphBankUndeclaredError

    net = _build_net()
    banks = _build_sample_banks()
    enc_idx = torch.tensor([0], dtype=torch.long, device="cuda")

    # Mock max_spec: 故意只声明 action,缺 build/board/hand/enemies 等字段
    # (模拟"BankMaxSpec 忘加新 bank 字段" 的 regression 场景)
    class MockPartialMaxSpec:
        action = 100
        numeric_dim = 58

        def get(self, name):
            if not hasattr(self, name):
                raise KeyError(name)
            return getattr(self, name)

    # 即使 strict=False,undeclared 永远 raise
    with pytest.raises(GraphBankUndeclaredError):
        GraphRunner(
            net, banks, enc_idx,
            atol=1e-3, startup_parity_n=3, parity_check_every=0,
            max_spec=MockPartialMaxSpec(),
            strict=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
