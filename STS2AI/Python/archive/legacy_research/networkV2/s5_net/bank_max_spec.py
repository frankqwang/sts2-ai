"""Banks padding 到固定 MAX_LEN 的规格定义。CUDA graph capture 需要固定 shape。

这些是 STS2 combat/noncombat 下 bank 的保守上限;超了会抛 BankOverflowError。
调大某个值不会破坏功能(只是浪费内存),调小可能在边缘场景 overflow。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BankMaxSpec:
    """每个 bank 的 (max_seq_len, max_numeric_dim)。

    numeric_dim 是 token 里 numeric features 数,通常 <=58(max_numeric_dim 默认值)。
    """
    # Shared banks
    build: int = 80           # player + deck cards + relics + potions
    inventory: int = 30       # relics/potions 详细
    economy: int = 20         # gold + shop items
    route: int = 40           # map nodes
    objective: int = 20       # boss + story goals
    forecast: int = 20        # lookahead encounters
    # Combat banks
    board: int = 32           # player + 手牌/敌人/牌堆 + draw distribution tokens
    mechanism: int = 20       # boss mechanics
    modifier: int = 30        # power modifiers
    power: int = 50           # all active powers (enemies * their powers)
    turn_prefix: int = 30     # turn history tokens
    combat_memory: int = 40   # multi-turn memory
    # Action bank
    action: int = 100         # legal actions (play_card * hand * enemies + end_turn + potions)

    # Numeric feature dimension (same for all banks, UnifiedNet.tokenizer default)
    numeric_dim: int = 58

    def get(self, bank_name: str) -> int:
        """按 bank name 取 max_len。unknown bank 抛异常(避免静默 fallback)。"""
        bn = bank_name.lower()
        if not hasattr(self, bn):
            raise KeyError(
                f"Unknown bank name '{bank_name}'. "
                f"Known: {sorted(self.__annotations__.keys())}. "
                f"加新 bank 需要在 BankMaxSpec 里声明 max_len."
            )
        val = getattr(self, bn)
        if not isinstance(val, int):
            raise KeyError(f"bank '{bank_name}' is not a length field")
        return int(val)


# 默认规格(保守上限)
DEFAULT_MAX_SPEC = BankMaxSpec()


class BankOverflowError(RuntimeError):
    """Bank seq_len 超过 MAX_SPEC。应该调大对应 max_len 或分析为什么突然爆量。"""
