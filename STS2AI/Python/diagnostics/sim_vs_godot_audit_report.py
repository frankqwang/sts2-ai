from __future__ import annotations
# --- wizardly cleanup 2026-04-08: tools/python subdir sys.path bootstrap ---
# Moved out of tools/python/ root; bootstrap below re-adds the parent dir so
# flat `from combat_nn import X` style imports still resolve.
import sys as _sys; from pathlib import Path as _Path  # noqa: E401,E702
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # noqa: E402
# --- end bootstrap ---

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception:
            continue
    return None


def _status_text(value: Any, *, non_blocking: bool = False) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "WARN" if non_blocking else "FAIL"
    return "N/A"


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _manifest_step(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    return ((manifest.get("steps") or {}).get(name) or {})


def _step_report_status(
    manifest: dict[str, Any],
    step_name: str,
    report: dict[str, Any] | None,
    *,
    report_key: str = "passed",
) -> Any:
    step = _manifest_step(manifest, step_name)
    if not step:
        return None
    if report is not None:
        return report.get(report_key)
    return bool(step.get("passed"))


def _coverage_lines(audit_report: dict[str, Any] | None) -> list[str]:
    if not audit_report:
        return ["Coverage data not found."]

    coverage = audit_report.get("coverage") or {}
    states = coverage.get("states") or {}
    transitions = coverage.get("transitions") or {}
    lines = [
        "| State | Coverage | Parity | Exact Save/Load | Stress | First Hit | Produced |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for state_name in (
        "map",
        "event",
        "rest_site",
        "shop",
        "treasure",
        "monster",
        "elite",
        "boss",
        "combat_pending",
        "combat_rewards",
        "card_reward",
        "card_select",
        "relic_select",
        "game_over",
    ):
        record = states.get(state_name) or {}
        first_hit = "N/A"
        if record.get("first_hit_seed"):
            first_hit = f"{record['first_hit_seed']}@{record.get('first_hit_step', '?')}"
        lines.append(
            "| {state} | {coverage} | {parity} | {exact} | {stress} | {first_hit} | {produced} |".format(
                state=state_name,
                coverage="Y" if record.get("coverage") else "N",
                parity="Y" if record.get("parity") else "N",
                exact="Y" if record.get("exact_save_load") else "N",
                stress="Y" if record.get("stress") else "N",
                first_hit=first_hit,
                produced="Y" if record.get("produced_by_full_run", True) else "N",
            )
        )

    lines.extend(
        [
            "",
            "| Transition | Count | Coverage | Stress | Required | First Hit |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for transition_key in (
        "event -> card_select",
        "event -> monster",
        "event -> relic_select",
        "event -> elite",
        "event -> boss",
        "combat_pending -> combat_rewards",
        "combat_pending -> map",
        "combat_pending -> game_over",
        "combat_rewards -> card_reward",
        "combat_rewards -> map",
        "combat_rewards -> event",
        "combat_rewards -> game_over",
        "treasure -> relic_select",
    ):
        raw_key = transition_key.replace(" -> ", "->")
        record = transitions.get(raw_key) or transitions.get(transition_key) or {}
        first_hit = "N/A"
        if record.get("first_hit_seed"):
            first_hit = f"{record['first_hit_seed']} ({record.get('first_hit_policy', '?')})"
        lines.append(
            "| {transition} | {count} | {coverage} | {stress} | {required} | {first_hit} |".format(
                transition=transition_key,
                count=record.get("count", 0),
                coverage=record.get("coverage_count", 0),
                stress=record.get("stress_count", 0),
                required="Y" if record.get("required") else "N",
                first_hit=first_hit,
            )
        )

    return lines


def build_markdown(run_root: Path) -> str:
    manifest = _load_json(run_root / "run_manifest.json") or {}
    quick_trace = _load_json(run_root / "quick" / "state_trace_parity_report.json")
    quick_audit = _load_json(run_root / "quick" / "audit_full_report.json")
    save_load = _load_json(run_root / "quick" / "save_load_report.json")
    nn_parity = _load_json(run_root / "quick" / "nn_backend_parity_report.json")
    display_parity = _load_json(run_root / "full" / "display_parity_report.json")
    boss_audit = _load_json(run_root / "full" / "boss_outcome_audit.json")
    reward_audit = _load_json(run_root / "full" / "reward_loop_audit.json")
    discover = _load_json(run_root / "full" / "discover_report.json")
    policy_rollout = _load_json(run_root / "full" / "policy_rollout_audit.json")
    baseline_combat_saveload = _load_json(run_root / "full" / "baseline_saveload_combat_parity.json")
    candidate_combat_saveload = _load_json(run_root / "full" / "candidate_saveload_combat_parity.json")
    training_semantic = _load_json(run_root / "full" / "training_semantic_audit.json")

    nn_summary = (nn_parity or {}).get("summary") or {}
    quick_steps_present = any(item is not None for item in (quick_trace, quick_audit, save_load, nn_parity)) or bool((manifest.get("steps") or {}).get("build"))
    quick_pass = None
    if quick_steps_present:
        quick_pass = bool(
            (quick_trace or {}).get("passed")
            and (quick_audit or {}).get("passed")
            and (save_load or {}).get("passed")
            and nn_summary.get("mismatch_seed_count", 1) == 0
            and bool((manifest.get("steps") or {}).get("build", {}).get("passed"))
        )

    full_report_states = {
        "boss_outcome_audit": _step_report_status(manifest, "full_boss_outcome_audit", boss_audit, report_key="pass"),
        "reward_loop_audit": _step_report_status(manifest, "full_reward_loop_audit", reward_audit, report_key="pass"),
        "discover": _step_report_status(manifest, "full_discover", discover, report_key="passed"),
        "policy_rollout_audit": _step_report_status(manifest, "full_policy_rollout_audit", policy_rollout, report_key="passed"),
        "training_semantic_audit": _step_report_status(manifest, "full_training_semantic_audit", training_semantic, report_key="passed"),
    }
    full_reports_present = any(value is not None for value in full_report_states.values())
    full_pass = None
    if full_reports_present:
        full_pass = all(value is True for value in full_report_states.values() if value is not None)

    display_pass = _step_report_status(manifest, "full_display_parity", display_parity, report_key="passed")

    lines: list[str] = [
        "# Sim vs Godot Audit Summary",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Run root: `{run_root}`",
        f"- Mode: `{manifest.get('mode', 'unknown')}`",
        f"- Baseline backend: `{manifest.get('baseline_backend', 'godot-http')}`",
        f"- Candidate backend: `{manifest.get('candidate_backend', 'headless-pipe')}`",
        "",
        "## Gate Verdicts",
        "",
        f"- Quick gate: **{_status_text(quick_pass)}**",
        f"- Full audit: **{_status_text(full_pass)}**",
        f"- Display parity (non-blocking): **{_status_text(display_pass, non_blocking=True)}**",
        "",
        "## Quick Gate",
        "",
        f"- Build: **{_status_text(_manifest_step(manifest, 'build').get('passed') if _manifest_step(manifest, 'build') else None)}**",
        f"- State trace parity: **{_status_text((quick_trace or {}).get('passed') if quick_trace is not None else None)}**",
        f"- Full parity audit: **{_status_text((quick_audit or {}).get('passed') if quick_audit is not None else None)}**",
        f"- Exact save/load: **{_status_text((save_load or {}).get('passed') if save_load is not None else None)}**",
        f"- NN backend parity: **{_status_text(nn_summary.get('mismatch_seed_count', 1) == 0 if nn_parity is not None else None)}**",
    ]

    if quick_trace:
        lines.extend(
            [
                "",
                "### State Trace Parity",
                "",
                f"- seed_count: `{quick_trace.get('seed_count', 0)}`",
                f"- mismatch_seed_count: `{quick_trace.get('mismatch_seed_count', 0)}`",
                f"- driver_mode: `{quick_trace.get('driver_mode', 'bidirectional')}`",
            ]
        )

    if save_load:
        summary = save_load.get("summary") or {}
        lines.extend(
            [
                "",
                "### Save/Load",
                "",
                f"- exact: `{summary.get('exact', 0)}`",
                f"- resumable: `{summary.get('resumable', 0)}`",
                f"- unsupported: `{summary.get('unsupported', 0)}`",
                f"- hard_failures: `{summary.get('hard_failures', 0)}`",
            ]
        )

    if nn_parity:
        lines.extend(
            [
                "",
                "### NN Parity",
                "",
                f"- seed_count: `{nn_summary.get('seed_count', 0)}`",
                f"- mismatch_seed_count: `{nn_summary.get('mismatch_seed_count', 0)}`",
                f"- mismatch_reasons: `{json.dumps(nn_summary.get('mismatch_reasons', {}), ensure_ascii=False)}`",
            ]
        )

    lines.extend(["", "## Full Audit", ""])
    if boss_audit:
        baseline = (boss_audit.get("summary") or {}).get("baseline") or {}
        candidate = (boss_audit.get("summary") or {}).get("candidate") or {}
        lines.extend(
            [
                "### Boss / MAX / UNK",
                "",
                f"- Result: **{_status_text(boss_audit.get('pass'))}**",
                f"- mismatch_count: `{(boss_audit.get('summary') or {}).get('mismatch_count', 0)}`",
                f"- baseline boss_unk_at_cap: `{baseline.get('boss_unk_at_cap', 0)}`",
                f"- candidate boss_unk_at_cap: `{candidate.get('boss_unk_at_cap', 0)}`",
                f"- baseline outcomes_at_cap: `{json.dumps(baseline.get('outcomes_at_cap', {}), ensure_ascii=False)}`",
                f"- candidate outcomes_at_cap: `{json.dumps(candidate.get('outcomes_at_cap', {}), ensure_ascii=False)}`",
                "",
            ]
        )
    elif _manifest_step(manifest, "full_boss_outcome_audit"):
        lines.extend(
            [
                "### Boss / MAX / UNK",
                "",
                f"- Result: **{_status_text(_step_report_status(manifest, 'full_boss_outcome_audit', None))}**",
                "- Report file missing; inspect step logs in `run_manifest.json`.",
                "",
            ]
        )

    if reward_audit:
        reward_summary = reward_audit.get("summary") or {}
        lines.extend(
            [
                "### Reward Chain",
                "",
                f"- Result: **{_status_text(reward_audit.get('pass'))}**",
                f"- mismatch_count: `{reward_summary.get('mismatch_count', 0)}`",
                f"- baseline_suspicious_chain_count: `{reward_summary.get('baseline_suspicious_chain_count', 0)}`",
                f"- candidate_suspicious_chain_count: `{reward_summary.get('candidate_suspicious_chain_count', 0)}`",
                "",
            ]
        )
    elif _manifest_step(manifest, "full_reward_loop_audit"):
        step = _manifest_step(manifest, "full_reward_loop_audit")
        lines.extend(
            [
                "### Reward Chain",
                "",
                f"- Result: **{_status_text(step.get('passed'))}**",
                f"- report_missing: `true`",
                f"- step_exit_code: `{step.get('exit_code')}`",
                "",
            ]
        )

    if training_semantic:
        comparison = training_semantic.get("comparison") or {}
        metrics = comparison.get("metrics") or {}
        lines.extend(
            [
                "### Training Semantic Canary",
                "",
                f"- Result: **{_status_text(training_semantic.get('passed'))}**",
                f"- failed_metrics: `{json.dumps(comparison.get('failed_metrics', []), ensure_ascii=False)}`",
                f"- avg_floor_tail_mean abs diff: `{_format_float((metrics.get('avg_floor_tail_mean') or {}).get('abs_diff'))}`",
                f"- boss_reach_tail_mean abs diff: `{_format_float((metrics.get('boss_reach_tail_mean') or {}).get('abs_diff'))}`",
                f"- act1_clear_tail_mean abs diff: `{_format_float((metrics.get('act1_clear_tail_mean') or {}).get('abs_diff'))}`",
                f"- avg_episode_steps rel diff: `{_format_float((metrics.get('avg_episode_steps') or {}).get('rel_diff'))}`",
                f"- avg_reward_hits rel diff: `{_format_float((metrics.get('avg_reward_hits') or {}).get('rel_diff'))}`",
                f"- avg_card_reward_hits rel diff: `{_format_float((metrics.get('avg_card_reward_hits') or {}).get('rel_diff'))}`",
                f"- MAX ratio abs diff: `{_format_float((metrics.get('max_ratio') or {}).get('abs_diff'))}`",
                f"- UNK ratio abs diff: `{_format_float((metrics.get('unk_ratio') or {}).get('abs_diff'))}`",
                f"- ERR ratio abs diff: `{_format_float((metrics.get('err_ratio') or {}).get('abs_diff'))}`",
                "",
                f"- baseline boss_reach: `{_format_pct(((training_semantic.get('baseline') or {}).get('replays') or {}).get('boss_reach_ratio'))}`",
                f"- candidate boss_reach: `{_format_pct(((training_semantic.get('candidate') or {}).get('replays') or {}).get('boss_reach_ratio'))}`",
                f"- baseline tags: `{json.dumps(((training_semantic.get('baseline') or {}).get('replays') or {}).get('tag_counts', {}), ensure_ascii=False)}`",
                f"- candidate tags: `{json.dumps(((training_semantic.get('candidate') or {}).get('replays') or {}).get('tag_counts', {}), ensure_ascii=False)}`",
                "",
            ]
        )
    elif _manifest_step(manifest, "full_training_semantic_audit"):
        step = _manifest_step(manifest, "full_training_semantic_audit")
        lines.extend(
            [
                "### Training Semantic Canary",
                "",
                f"- Result: **{_status_text(step.get('passed'))}**",
                f"- report_missing: `true`",
                f"- step_exit_code: `{step.get('exit_code')}`",
                "",
            ]
        )

    if policy_rollout:
        comparison = policy_rollout.get("comparison") or {}
        metrics = comparison.get("metrics") or {}
        lines.extend(
            [
                "### Policy Rollout Audit",
                "",
                f"- Result: **{_status_text(policy_rollout.get('passed'))}**",
                f"- failed_metrics: `{json.dumps(comparison.get('failed_metrics', []), ensure_ascii=False)}`",
                f"- per_seed_mismatch_count: `{comparison.get('per_seed_mismatch_count', 0)}`",
                f"- avg_floor abs diff: `{_format_float((metrics.get('avg_floor') or {}).get('abs_diff'))}`",
                f"- boss_reach_rate abs diff: `{_format_float((metrics.get('boss_reach_rate') or {}).get('abs_diff'))}`",
                f"- act1_clear_rate abs diff: `{_format_float((metrics.get('act1_clear_rate') or {}).get('abs_diff'))}`",
                f"- avg_steps rel diff: `{_format_float((metrics.get('avg_steps') or {}).get('rel_diff'))}`",
                "",
            ]
        )
    elif _manifest_step(manifest, "full_policy_rollout_audit"):
        step = _manifest_step(manifest, "full_policy_rollout_audit")
        lines.extend(
            [
                "### Policy Rollout Audit",
                "",
                f"- Result: **{_status_text(step.get('passed'))}**",
                f"- report_missing: `true`",
                f"- step_exit_code: `{step.get('exit_code')}`",
                "",
            ]
        )

    if baseline_combat_saveload or candidate_combat_saveload:
        def _combat_parity_line(label: str, payload: dict[str, Any] | None) -> list[str]:
            if not payload:
                return [f"- {label}: `N/A`"]
            summary = payload.get("summary") or {}
            return [
                (
                    f"- {label}: exact=`{summary.get('exact', 0)}` "
                    f"resumable=`{summary.get('resumable', 0)}` "
                    f"diverged=`{summary.get('diverged', 0)}` "
                    f"hand_order_rate=`{_format_pct(summary.get('hand_order_preserved_rate'))}` "
                    f"legal_rate=`{_format_pct(summary.get('legal_actions_preserved_rate'))}` "
                    f"mcts_feasibility=`{summary.get('mcts_feasibility', 'N/A')}`"
                )
            ]

        lines.extend(
            [
                "### Combat Save/Load Parity (advisory)",
                "",
                *_combat_parity_line("baseline", baseline_combat_saveload),
                *_combat_parity_line("candidate", candidate_combat_saveload),
                "",
            ]
        )

    lines.extend(["## Coverage Matrix", ""])
    lines.extend(_coverage_lines(quick_audit))

    if discover:
        lines.extend(
            [
                "",
                "## Discover",
                "",
                f"- Result: **{_status_text(discover.get('passed'))}**",
                f"- runtime_failures: `{len(discover.get('runtime_failures', []))}`",
                f"- required_state_targets: `{json.dumps(discover.get('required_state_targets', []), ensure_ascii=False)}`",
                f"- optional_state_targets: `{json.dumps(discover.get('optional_state_targets', []), ensure_ascii=False)}`",
                f"- required_transition_targets: `{json.dumps(discover.get('required_transition_targets', []), ensure_ascii=False)}`",
                f"- optional_transition_targets: `{json.dumps(discover.get('optional_transition_targets', []), ensure_ascii=False)}`",
            ]
        )
        found_states = discover.get("found_states") or {}
        found_transitions = discover.get("found_transitions") or {}
        if found_states:
            lines.extend(["", "### Found States", ""])
            for state_name, hit in sorted(found_states.items()):
                lines.append(f"- `{state_name}`: `{hit.get('seed')}@{hit.get('step')}`")
        if found_transitions:
            lines.extend(["", "### Found Transitions", ""])
            for transition_name, hit in sorted(found_transitions.items()):
                lines.append(f"- `{transition_name}`: `{hit.get('seed')}@{hit.get('step')}`")
        static_notes = discover.get("static_notes") or {}
        if static_notes:
            lines.extend(["", "### Static Notes", ""])
            for key, value in sorted(static_notes.items()):
                lines.append(f"- `{key}`: {value}")
    elif _manifest_step(manifest, "full_discover"):
        step = _manifest_step(manifest, "full_discover")
        lines.extend(
            [
                "",
                "## Discover",
                "",
                f"- Result: **{_status_text(step.get('passed'))}**",
                f"- report_missing: `true`",
                f"- step_exit_code: `{step.get('exit_code')}`",
            ]
        )

    lines.extend(
        [
            "",
        "## Report Paths",
        "",
        "- `quick/state_trace_parity_report.json`",
        "- `quick/audit_full_report.json`",
        "- `quick/save_load_report.json`",
        "- `quick/nn_backend_parity_report.json`",
            "- `full/display_parity_report.json`",
            "- `full/boss_outcome_audit.json`",
            "- `full/reward_loop_audit.json`",
            "- `full/discover_report.json`",
            "- `full/policy_rollout_audit.json`",
            "- `full/baseline_saveload_combat_parity.json`",
            "- `full/candidate_saveload_combat_parity.json`",
            "- `full/training_semantic_audit.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown summary for sim-vs-Godot audit runs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output or (args.run_root / "verification_summary.md")
    markdown = build_markdown(args.run_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
