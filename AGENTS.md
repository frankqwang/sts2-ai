# AGENTS.md

## Global Repo Rules
- 文档用中文
- Local clickable file links in Codex responses must use a leading `/` before the absolute Windows path.
- Correct format: `[train_hybrid.py](/C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:172)`
- Wrong format: `[train_hybrid.py](C:/Users/Administrator/Desktop/sts2Raw2/STS2AI/Python/train_hybrid.py:172)`
- Prefer including a line number when referencing code.

- Except for `README` files, project documentation should be placed under `STS2AI/docs`.
- New handoff notes, protocols, working notes, and similar docs should not be added at repo root.

不要修改src下的任务源码，如果一定要改，请告知我。这是反编译出来的，我们的代码在sts2ai下面；
避免中文编码问题，各种工具里能用尽量用utf-8