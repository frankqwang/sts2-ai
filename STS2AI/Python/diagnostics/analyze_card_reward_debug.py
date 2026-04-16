"""Analyze card-reward selection sources, overrides, and boss-specific pick distributions."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_iteration_replays import SkadaNameResolver
from analyze_training_window import _load_summaries


def _top(counter: Counter[str], resolver: SkadaNameResolver, *, kind: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, count in counter.most_common(limit):
        if kind == "card":
            display = resolver.card(name)
        else:
            display = name
        rows.append({"name": name, "display": display, "count": int(count)})
    return rows


def build_report(summaries: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    resolver = SkadaNameResolver()
    source_counter: Counter[str] = Counter()
    boss_counter: Counter[str] = Counter()
    selected_counter: Counter[str] = Counter()
    override_selected_counter: Counter[str] = Counter()
    boss_selected: dict[str, Counter[str]] = defaultdict(Counter)
    raw_top_missed = 0
    total = 0
    total_selected_bonus = 0.0

    for summary in summaries:
        for reward in summary.get("card_rewards") or []:
            debug = reward.get("decision_debug")
            if not isinstance(debug, dict):
                continue
            total += 1
            source = str(debug.get("source") or "none")
            boss_token = str(debug.get("boss_token") or "unknown")
            source_counter[source] += 1
            boss_counter[boss_token] += 1
            choices = debug.get("choices") or []
            if not isinstance(choices, list) or not choices:
                continue
            selected = next((choice for choice in choices if bool(choice.get("selected"))), None)
            if selected is None:
                continue
            selected_name = str(selected.get("label") or "unknown")
            selected_counter[selected_name] += 1
            boss_selected[boss_token][selected_name] += 1
            total_selected_bonus += float(selected.get("boss_bonus") or 0.0)

            raw_top = max(
                choices,
                key=lambda item: (float(item.get("raw_logit") or 0.0), float(item.get("prob") or 0.0)),
            )
            if str(raw_top.get("label") or "") != selected_name:
                raw_top_missed += 1
                override_selected_counter[selected_name] += 1

    by_boss = {}
    for boss_token, counter in sorted(boss_selected.items()):
        by_boss[boss_token] = _top(counter, resolver, kind="card", limit=top_k)

    return {
        "decision_count": int(total),
        "raw_top_overridden_count": int(raw_top_missed),
        "raw_top_overridden_rate": round(raw_top_missed / max(1, total), 4),
        "avg_selected_boss_bonus": round(total_selected_bonus / max(1, total), 4),
        "source_top": _top(source_counter, resolver, kind="text", limit=top_k),
        "boss_top": _top(boss_counter, resolver, kind="text", limit=top_k),
        "selected_card_top": _top(selected_counter, resolver, kind="card", limit=top_k),
        "override_selected_card_top": _top(override_selected_counter, resolver, kind="card", limit=top_k),
        "selected_by_boss": by_boss,
    }


def render_markdown(report: dict[str, Any], *, iter_start: int, iter_end: int) -> str:
    lines = [
        f"# 卡奖决策解释汇总（{iter_start}-{iter_end}）",
        "",
        f"- 决策总数: `{report.get('decision_count', 0)}`",
        f"- 原始 PPO top1 被 rerank 改写次数: `{report.get('raw_top_overridden_count', 0)}` "
        f"({float(report.get('raw_top_overridden_rate', 0.0)) * 100:.2f}%)",
        f"- 选中动作平均 boss bonus: `{float(report.get('avg_selected_boss_bonus', 0.0)):.4f}`",
        "",
        "## 决策来源 Top",
    ]
    for row in report.get("source_top") or []:
        lines.append(f"- `{row['name']}`: `{row['count']}`")
    lines.extend(["", "## Boss Top"])
    for row in report.get("boss_top") or []:
        lines.append(f"- `{row['name']}`: `{row['count']}`")
    lines.extend(["", "## 选中卡牌 Top"])
    for row in report.get("selected_card_top") or []:
        lines.append(f"- `{row['display']}` (`{row['name']}`): `{row['count']}`")
    lines.extend(["", "## 被 rerank 改写后选中的卡牌 Top"])
    for row in report.get("override_selected_card_top") or []:
        lines.append(f"- `{row['display']}` (`{row['name']}`): `{row['count']}`")
    lines.append("")
    for boss_token, rows in (report.get("selected_by_boss") or {}).items():
        lines.append(f"## Boss `{boss_token}` 的选中卡牌 Top")
        for row in rows:
            lines.append(f"- `{row['display']}` (`{row['name']}`): `{row['count']}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze card reward decision debug summaries.")
    parser.add_argument("training_dir", type=Path)
    parser.add_argument("--iter-start", type=int, required=True)
    parser.add_argument("--iter-end", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    summaries = _load_summaries(args.training_dir, args.iter_start, args.iter_end)
    report = build_report(summaries, top_k=max(1, args.top_k))

    out_dir = args.training_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"card_reward_debug_from_{args.iter_start:05d}_to_{args.iter_end:05d}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report, iter_start=args.iter_start, iter_end=args.iter_end), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
