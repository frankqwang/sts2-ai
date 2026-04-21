from __future__ import annotations

"""显式关闭搜索链路时使用的空 search backend。

这个 backend 的职责不是“提供一个更弱的搜索”，而是明确表达：

- 当前主线不做搜索打标
- 当前 collect 不该依赖搜索动作
- loop / trainer 仍可复用统一接口，不必为 `None` 特判

因此 `label_request(...)` 始终返回空 `SearchLabel`，不会生成 policy /
best_action / trace。若调用方真的需要搜索结果，应显式切到实验搜索模式。
"""

from ..domain import SearchLabel, SearchRequest


class NoopSearchBackend:
    """主线关闭搜索时的占位 backend。"""

    def label_request(self, request: SearchRequest, runtime_factory=None, seed: str | None = None, policy=None) -> SearchLabel:
        return SearchLabel(
            metadata={
                "search_backend": "NoopSearchBackend",
                "search_mode": "disabled",
            }
        )

