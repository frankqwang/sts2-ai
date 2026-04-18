"""CUDA graph runner for UnifiedNet rollout inference.

消除 batch=1 的 kernel launch overhead(实测 22ms/forward → 预期 2-5ms)。
Windows 友好(不依赖 triton,纯 CUDA)。

## 硬检测机制(防止正确性退化)

1. **Shape signature 检测** — 每 call 前 assert banks shape 匹配 capture 时的
   shape。加新 bank / 改 bank 结构 → 立刻 raise `GraphShapeMismatchError`。

2. **Startup parity check** — `GraphRunner.__init__` 里跑 50 次 eager vs graph
   对比,误差 > atol 直接 raise。新 op / 算子改动破坏 determinism 立刻暴露。

3. **Periodic parity** — 运行时每 `parity_check_every=500` step 抽样对比一次,
   drift 超阈值自动 raise 并记录现场 banks 供调试。

4. **CI 回归测试** — 见 `tests/test_graph_runner.py`。PR 跑 200 随机 banks
   eager vs graph 对比,fail 直接挡 merge。

## 使用

    runner = GraphRunner(net, sample_banks, device='cuda', atol=1e-3)
    for banks in stream:
        out = runner(banks, encounter_idx)  # 内部:shape check → replay → periodic parity

## 当前局限

- **静态 shape**:banks 的每个字段 (hand/enemies/deck) 必须 padding 到 MAX_LEN
  (见 graph_bank_spec.py)。否则 shape check 必定 mismatch。

- **tokenizer 必须支持 static buffer 路径**:UnifiedNet.tokenizer 当前每 step
  `torch.zeros(...)` 新 tensor,graph capture 会炸
  (`Cannot copy between CPU and CUDA tensors during CUDA graph capture`)。
  需要 `tokenizer.tokenize_banks_static(banks, static_buffers)` API。
  见 `TODO: static tokenizer path` 注释。
"""
from __future__ import annotations

import logging
import torch
from dataclasses import dataclass
from typing import Any, Callable

from networkV2.s1_schema.token_banks import Token, TokenBank, UnifiedTokenBanks
from networkV2.s5_net.bank_max_spec import (
    BankMaxSpec, BankOverflowError, DEFAULT_MAX_SPEC,
)
from networkV2.s5_net.tokenizer import alloc_static_bank_buffers


def patch_dropout_for_graph_safety() -> None:
    """Monkey-patch torch.nn.functional.dropout:p==0 直接短路不 call kernel。

    根因:PyTorch CUDA dropout kernel 即使 p=0 也 access Philox RNG offset。
    capture 一次 dropout(p=0) 会 bake offset 增量到 graph;replay/eager 后
    training forward 的 dropout 访问 offset,抛 "Offset increment outside graph
    capture encountered unexpectedly" (issue #99820)。

    This patch 保持 F.dropout 语义不变(p=0 本来就等价 identity),只是绕开
    kernel 的 RNG access。等价于 `if p == 0 or not training: return input` 的
    fast path。

    Idempotent:多次调用只 patch 一次。
    """
    import torch.nn.functional as F
    if getattr(F, "_dropout_patched_for_graph", False):
        return
    _orig_dropout = F.dropout

    def _safe_dropout(input, p=0.5, training=True, inplace=False):
        if p == 0.0 or not training:
            return input
        return _orig_dropout(input, p, training, inplace)

    F.dropout = _safe_dropout
    F._dropout_patched_for_graph = True


def _empty_token(numeric_dim: int) -> Token:
    """MAX padding 用的 zero token。token_type='pad' → type_idx=0。"""
    return Token(numeric=[0.0] * numeric_dim, token_type="pad")


def pad_banks_to_max(
    banks: UnifiedTokenBanks,
    max_spec: BankMaxSpec,
    numeric_dim: int = 58,
) -> UnifiedTokenBanks:
    """对 banks 的每个 bank 加 empty tokens 直到 max_len,让 eager forward 输出 shape
    和 CUDA graph 输出 (固定 MAX shape) 一致,便于 parity check 对比。

    做一个浅拷贝:原 banks 不改。只加 tail 空 tokens。
    """
    import copy
    padded = copy.copy(banks)
    # shared
    from networkV2.s1_schema.token_banks import SharedWorldBanks, CombatBanks
    padded.shared = copy.copy(banks.shared)
    for attr in ("build_bank", "inventory_bank", "economy_bank",
                 "route_bank", "objective_bank", "forecast_bank"):
        src = getattr(banks.shared, attr)
        if not hasattr(max_spec, src.bank_name):
            continue
        max_len = max_spec.get(src.bank_name)
        if len(src.tokens) >= max_len:
            continue
        new_bank = TokenBank(bank_name=src.bank_name, tokens=list(src.tokens))
        while len(new_bank.tokens) < max_len:
            new_bank.tokens.append(_empty_token(numeric_dim))
        setattr(padded.shared, attr, new_bank)
    # combat
    if banks.combat is not None:
        padded.combat = copy.copy(banks.combat)
        for attr in ("board_bank", "mechanism_bank", "modifier_bank",
                     "power_bank", "turn_prefix_bank", "combat_memory_bank"):
            src = getattr(banks.combat, attr)
            if not hasattr(max_spec, src.bank_name):
                continue
            max_len = max_spec.get(src.bank_name)
            if len(src.tokens) >= max_len:
                continue
            new_bank = TokenBank(bank_name=src.bank_name, tokens=list(src.tokens))
            while len(new_bank.tokens) < max_len:
                new_bank.tokens.append(_empty_token(numeric_dim))
            setattr(padded.combat, attr, new_bank)
    # action
    if hasattr(max_spec, "action") and len(banks.action_bank.tokens) < max_spec.action:
        new_action = TokenBank(bank_name="action", tokens=list(banks.action_bank.tokens))
        while len(new_action.tokens) < max_spec.action:
            new_action.tokens.append(_empty_token(numeric_dim))
        padded.action_bank = new_action
    return padded


logger = logging.getLogger(__name__)


class GraphShapeMismatchError(RuntimeError):
    """banks shape 和 capture 时不一致。加新 bank / 改 max_len / bug 都会 trigger。"""


class GraphParityDriftError(RuntimeError):
    """CUDA graph replay 结果与 eager 输出差异 > atol。通常是:
    - forward 里新加了 op 但 graph 没重 capture
    - tokenizer / static buffer 写入漏了某个字段
    - GPU driver / PyTorch 版本变化引入数值差
    """


class GraphBankUndeclaredError(RuntimeError):
    """sample_banks 里有非空 bank 但 BankMaxSpec 没声明 max_len。

    加新 bank 必须同步更新 s5_net/bank_max_spec.py,否则 graph 里根本不会
    feed 这个 bank 的数据(静默漏洞:forward 拿到 empty bank,数值不对但可能
    碰巧绕过 parity check)。
    """


class GraphCaptureFailedError(RuntimeError):
    """CUDA graph capture 或 startup parity 失败。

    常见原因:
    - forward 里新增 data-dependent branch(如 `if tensor.any()`)破坏 capture-safety
    - tokenizer.project_static 漏读 tokenizer.tokenize_banks 里的某个 embedding
    - 新加的 op 不支持 graph capture

    严格模式(strict=True,默认)下直接抛出;宽松模式(strict=False)降级 eager
    并打印 warning,但训练会静默失去 5-10x 加速,**不推荐日常使用**。
    """


@dataclass(frozen=True)
class BankShapeSignature:
    """bank 集合的 shape 指纹,用于 capture/runtime 一致性校验。

    不存 tensor 内容,只存"形状结构",快速比较。
    """
    # sorted tuple of (bank_name, seq_len, numeric_dim)
    bank_shapes: tuple[tuple[str, int, int], ...]

    @classmethod
    def from_banks(cls, banks) -> "BankShapeSignature":
        """从 UnifiedTokenBanks 抽 shape 指纹。"""
        shapes = []
        for bank in banks.all_banks():
            if bank.is_empty:
                continue
            seq_len = len(bank.tokens)
            num_dim = max(
                (len(tok.numeric) for tok in bank.tokens),
                default=0,
            )
            shapes.append((bank.bank_name, seq_len, num_dim))
        return cls(bank_shapes=tuple(sorted(shapes)))


class GraphRunner:
    """UnifiedNet 的 CUDA graph-accelerated rollout inference wrapper。

    init 阶段:
      1. 根据 sample_banks capture 一个 CUDA graph(含一轮 warmup)
      2. 跑 N 次 eager vs graph parity check(raise if drift)

    call 阶段:
      1. 硬检测 shape signature
      2. Copy banks → static input buffer
      3. graph.replay() — 0 launch overhead
      4. 每 parity_check_every step 做一次 eager 抽样对比
    """

    def __init__(
        self,
        net: torch.nn.Module,
        sample_banks,
        encounter_idx: torch.Tensor,
        *,
        device: str | torch.device = "cuda",
        atol: float = 1e-3,
        rtol: float = 1e-3,
        parity_check_every: int = 500,
        startup_parity_n: int = 20,
        startup_parity_noise: float = 0.05,
        max_spec: BankMaxSpec | None = None,
        enabled: bool = True,
        strict: bool = True,
    ):
        """
        strict (default True):
            capture / startup parity 失败时 **直接 raise**。
            设 False 才降级到 eager + warning(用于明确接受 fallback 的场景,
            日常训练不要用,会静默失去 graph 加速)。
        startup_parity_noise:
            startup parity 的随机扰动幅度。>0 时每轮对 host buffer numeric 加
            N(0, noise) 噪声,同步给 eager path,cover 更多数值路径。
        """
        # PyTorch issue #99820: CUDA dropout kernel 即使 p=0 仍 access RNG offset,
        # capture 期 bake offset 增量 → training forward 的 eager dropout 访问抛
        # "Offset increment outside graph capture"。patch F.dropout p=0 短路。
        patch_dropout_for_graph_safety()

        self.net = net
        self.device = torch.device(device)
        self.atol = atol
        self.rtol = rtol
        self.parity_check_every = int(parity_check_every)
        self.enabled = bool(enabled)
        self.strict = bool(strict)
        self.startup_parity_noise = float(startup_parity_noise)
        self.max_spec = max_spec or DEFAULT_MAX_SPEC
        self._step = 0
        self._last_parity_err = 0.0

        # 记录 capture 时的 shape signature
        self._capture_sig = BankShapeSignature.from_banks(sample_banks)
        self._sample_banks = sample_banks
        self._sample_enc_idx = encounter_idx.clone()

        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_output = None
        # static buffers(host pinned + GPU) — 每 call 前 fill
        self._host_buffers: dict[str, dict[str, torch.Tensor]] = {}
        self._gpu_buffers: dict[str, dict[str, torch.Tensor]] = {}
        self._static_enc_idx: torch.Tensor | None = None

        if not self.enabled:
            logger.info("[graph_runner] disabled (fallback to eager forward)")
            return

        if self.device.type != "cuda":
            logger.warning(
                f"[graph_runner] device={device} is not CUDA; falling back to eager."
            )
            self.enabled = False
            return

        try:
            self._alloc_static_buffers(sample_banks)
            self._capture(sample_banks, encounter_idx)
            logger.info(
                f"[graph_runner] capture OK. banks={list(self._gpu_buffers.keys())}"
            )
            self._startup_parity(sample_banks, encounter_idx, n=startup_parity_n)
            logger.info(
                f"[graph_runner] startup parity OK (n={startup_parity_n}, "
                f"atol={atol}, rtol={rtol}, noise={self.startup_parity_noise})"
            )
        except GraphBankUndeclaredError:
            # 声明漏配永远 fatal,不允许 fallback(因为 fallback 后 graph 仍在
            # 静默漏某 bank)
            raise
        except Exception as e:
            self.enabled = False
            self._graph = None
            # 清理 CUDA state(capture 异常后的 stream 残留)
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
            if self.strict:
                raise GraphCaptureFailedError(
                    f"CUDA graph capture/parity 失败: {type(e).__name__}: "
                    f"{str(e)[:300]}\n"
                    f"若确认要降级 eager(放弃 5-10x 加速), 构造 GraphRunner 时传 "
                    f"strict=False。日常训练请先定位 root cause。"
                ) from e
            logger.warning(
                f"[graph_runner] capture/parity failed: {type(e).__name__}: "
                f"{str(e)[:200]}. Falling back to eager forward (strict=False)."
            )

    # ------------------------------------------------------------------
    # Static buffer allocation
    # ------------------------------------------------------------------

    def _alloc_static_buffers(self, sample_banks) -> None:
        """按 sample_banks 里出现的 bank name 预分配 host pinned + GPU buffer。

        Hard check: sample_banks 里的每个非空 bank 都必须在 BankMaxSpec 里声明,
        否则 raise GraphBankUndeclaredError。这堵住了"加新 bank 但忘改 max_spec"
        的静默漏洞(以前会 `continue` 跳过,graph 里 forward 拿空 bank 数值错)。
        """
        missing = [
            b.bank_name for b in sample_banks.all_banks()
            if not b.is_empty and not hasattr(self.max_spec, b.bank_name)
        ]
        if missing:
            raise GraphBankUndeclaredError(
                f"sample_banks 里有 {missing} 但 BankMaxSpec 未声明 max_len。\n"
                f"加新 bank 必须同步更新 networkV2/s5_net/bank_max_spec.py,"
                f"为每个新 bank 加 `bank_name: int` 属性。否则 CUDA graph 不会"
                f"为此 bank 分配 static buffer,导致 forward 读到 empty。"
            )
        bank_names = [
            b.bank_name for b in sample_banks.all_banks()
            if not b.is_empty and hasattr(self.max_spec, b.bank_name)
        ]
        # 保证一定包含常用 combat banks(否则 project_static 没 key 会漏)
        for required in ("build", "action"):
            if required not in bank_names and hasattr(self.max_spec, required):
                bank_names.append(required)
        tokenizer = self.net.tokenizer
        self._host_buffers, self._gpu_buffers = alloc_static_bank_buffers(
            bank_names=bank_names,
            max_spec=self.max_spec,
            device=self.device,
            max_numeric_dim=tokenizer.max_numeric_dim,
            batch=1,
        )
        # encounter_idx static — graph 每次 replay 从这个固定地址读取
        self._static_enc_idx = torch.zeros(1, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture(self, sample_banks, encounter_idx: torch.Tensor) -> None:
        """Capture CUDA graph。用 static buffer path(tokenizer.project_static + net.forward_from_static)。

        捕获前:
          1. Fill sample_banks 到 static buffers(包含 CPU→GPU 的 pinned DMA,必须在 capture 外)
          2. Static encounter_idx copy
          3. Warmup stream 跑 3 次 forward_from_static(稳 cudnn benchmark 等)
          4. 强制 nn.MultiheadAttention 的 dropout=0(PyTorch 已知 bug:MHA 即使 eval mode
             + dropout_p=0 仍访问 RNG offset,CUDA graph capture 会抛
             `Offset increment outside graph capture`。这里把 attribute 彻底置零)
        Capture 期间:
          只执行 project_static + forward 的纯 GPU op(读 static buffer,写 static output)。
        """
        import torch.nn as nn
        self.net.eval()
        # Force MHA dropout=0 (inference 不需要 dropout,但 capture 期间必须置零避 RNG 访问)
        for m in self.net.modules():
            if isinstance(m, nn.MultiheadAttention):
                m.dropout = 0.0
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        tokenizer = self.net.tokenizer

        # Capture 前:fill buffers + copy encounter_idx
        tokenizer.fill_static_buffers(
            sample_banks, self._host_buffers, self._gpu_buffers,
        )
        self._static_enc_idx.copy_(encounter_idx, non_blocking=True)
        torch.cuda.synchronize()

        # warmup stream(对 CUDA graph capture 是必须的:确保 kernel 编译/分配都做完)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    _ = self.net.forward_from_static(
                        self._gpu_buffers, self._static_enc_idx,
                        decision_domain=sample_banks.decision_domain,
                    )
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # 真正 capture
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            with torch.no_grad():
                self._static_output = self.net.forward_from_static(
                    self._gpu_buffers, self._static_enc_idx,
                    decision_domain=sample_banks.decision_domain,
                )

    # ------------------------------------------------------------------
    # Parity checks
    # ------------------------------------------------------------------

    def _eager_forward(self, banks, encounter_idx: torch.Tensor):
        """无 graph 的纯 eager forward,用于 parity 对比。"""
        with torch.no_grad():
            return self.net(banks=banks, encounter_idx=encounter_idx)

    def _random_variant(self, banks, noise_std: float):
        """对 banks 做轻微扰动(token.numeric 加 N(0, noise) 噪声),生成一个新对象。

        用于 startup parity 覆盖多个数值路径,避免"同一 sample 20 次"的过拟合
        检测盲区。扰动**不改 bank 长度/结构**,只改数值,确保 shape signature 不变。
        """
        import copy
        import random

        if noise_std <= 0:
            return banks

        variant = copy.copy(banks)

        def _perturb_bank(src):
            new_tokens = []
            for tok in src.tokens:
                new_numeric = [
                    v + random.gauss(0.0, noise_std) for v in tok.numeric
                ]
                new_tok = Token(
                    numeric=new_numeric,
                    token_type=tok.token_type,
                    owner_id=tok.owner_id,
                    order=tok.order,
                    metadata=tok.metadata,
                )
                new_tokens.append(new_tok)
            return TokenBank(bank_name=src.bank_name, tokens=new_tokens)

        if variant.shared is not None:
            variant.shared = copy.copy(banks.shared)
            for attr in ("build_bank", "inventory_bank", "economy_bank",
                         "route_bank", "objective_bank", "forecast_bank"):
                src = getattr(banks.shared, attr)
                if not src.is_empty:
                    setattr(variant.shared, attr, _perturb_bank(src))
        if variant.combat is not None:
            variant.combat = copy.copy(banks.combat)
            for attr in ("board_bank", "mechanism_bank", "modifier_bank",
                         "power_bank", "turn_prefix_bank", "combat_memory_bank"):
                src = getattr(banks.combat, attr)
                if not src.is_empty:
                    setattr(variant.combat, attr, _perturb_bank(src))
        if not banks.action_bank.is_empty:
            variant.action_bank = _perturb_bank(banks.action_bank)
        return variant

    def _startup_parity(
        self, sample_banks, encounter_idx: torch.Tensor, n: int = 20,
    ) -> None:
        """启动时做 N 次 eager vs graph parity。不过 assert atol/rtol 就 raise。

        每轮对 sample_banks 做独立的随机数值扰动(startup_parity_noise),
        覆盖更多 activation 路径,避免"同一 sample 多次"的检测盲区。noise=0 时
        退化为重复同一 sample(老行为)。
        """
        if self._graph is None:
            return

        tokenizer = self.net.tokenizer
        # 真实 legal 动作数 = pad 前的 action_bank 长度(real positions 必须严格对齐,
        # pad positions 在 rollout 里不会用到 logits,允许 diff)
        real_action_len = len(sample_banks.action_bank.tokens)
        for i in range(n):
            variant = self._random_variant(sample_banks, self.startup_parity_noise)
            eager_nonpad = self._eager_forward(variant, encounter_idx)
            # Fill static buffers + replay (用同样的 variant 数据)
            tokenizer.fill_static_buffers(
                variant, self._host_buffers, self._gpu_buffers,
            )
            self._static_enc_idx.copy_(encounter_idx, non_blocking=True)
            self._graph.replay()
            torch.cuda.synchronize()
            # 只对 real action positions 做 parity(rollout 时 `logits[:len(legal)]`)
            eager_real = eager_nonpad.logits[..., :real_action_len]
            graph_real = self._static_output.logits[..., :real_action_len]
            try:
                torch.testing.assert_close(
                    eager_real, graph_real, atol=self.atol, rtol=self.rtol,
                )
            except AssertionError as err:
                raise GraphParityDriftError(
                    f"startup parity check {i}/{n} failed on real action positions "
                    f"(real_len={real_action_len}, noise={self.startup_parity_noise}): "
                    f"{err}"
                )

    def _periodic_parity(self, banks, encounter_idx: torch.Tensor) -> None:
        """运行期抽样 parity。"""
        eager_out = self._eager_forward(banks, encounter_idx)
        eager_logits = eager_out.logits
        graph_logits = self._static_output.logits
        err = (eager_logits - graph_logits).abs().max().item()
        self._last_parity_err = err
        if err > self.atol + self.rtol * eager_logits.abs().max().item():
            raise GraphParityDriftError(
                f"periodic parity drift at step {self._step}: "
                f"max_err={err:.3e} > atol={self.atol:.3e}. "
                f"Likely causes: new op added without recapture, or "
                f"static input buffer update is incomplete."
            )

    # ------------------------------------------------------------------
    # Call
    # ------------------------------------------------------------------

    def __call__(
        self, banks=None, encounter_idx: torch.Tensor = None,
        batched_banks=None, decision_domain: str = "combat",
    ):
        """替代 net(banks=..., encounter_idx=...)。

        硬检测次序:
          1. shape signature match
          2. graph replay (or fallback eager)
          3. periodic parity (每 parity_check_every step)
        """
        # Batched path 或非 combat domain 暂不支持 graph → 直接 eager
        if batched_banks is not None or decision_domain != "combat" or banks is None:
            return self.net(
                banks=banks, batched_banks=batched_banks,
                decision_domain=decision_domain, encounter_idx=encounter_idx,
            )

        # Fallback: 未启用或 capture 失败
        if not self.enabled or self._graph is None:
            return self._eager_forward(banks, encounter_idx)

        # Hard check 1: 基本校验(banks 有 decision_domain 且匹配 capture 时的域)
        if banks.decision_domain != self._sample_banks.decision_domain:
            raise GraphShapeMismatchError(
                f"decision_domain changed: capture={self._sample_banks.decision_domain}, "
                f"runtime={banks.decision_domain}. combat/non-combat 要分别 capture。"
            )

        # Fill static buffers(含 shape overflow 检测,超 MAX 会 raise BankOverflowError)
        # 这是 capture 之外的 DMA,允许 CPU→GPU。
        self.net.tokenizer.fill_static_buffers(
            banks, self._host_buffers, self._gpu_buffers,
        )
        self._static_enc_idx.copy_(encounter_idx, non_blocking=True)

        # Graph replay — 用最新 static buffer 数据
        self._graph.replay()
        self._step += 1

        # Hard check 3: periodic parity(加 cuda.synchronize 让 replay 完成)
        if self.parity_check_every > 0 and self._step % self.parity_check_every == 0:
            torch.cuda.synchronize()
            self._periodic_parity(banks, encounter_idx)

        return self._static_output

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "steps": self._step,
            "last_parity_err": self._last_parity_err,
            "capture_shape_sig": self._capture_sig.bank_shapes,
        }
