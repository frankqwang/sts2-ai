#!/usr/bin/env python3
"""Training health monitor — detects anomalies in training metrics.

Reusable across all training modes (PPO, MCTS, Hybrid).
Operates on metrics JSONL entries (list of dicts).

Usage:
    # Offline analysis
    python training_health.py artifacts/hybrid_training/hybrid_*/metrics.jsonl
    python training_health.py artifacts/rl_training/ppo_*/metrics.jsonl

    # Programmatic use in training loops
    from training_health import TrainingHealthMonitor
    monitor = TrainingHealthMonitor()
    alerts = monitor.check_all(entries)
    for a in alerts:
        logger.warning("HEALTH: %s", a)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Rollout quality gate
# ---------------------------------------------------------------------------

def quality_check(results, iteration, quarantine_dir=None):
    """Validate a batch of episodes before using them for training.

    Moved here from the now-archived `train_rl.py` (pre-hybrid trainer) so
    the smoke tests still have a home for it. `train_hybrid.py` does not
    use this helper — it has its own guards — but the smoke suite exercises
    it as a unit test for the quality-gate logic.

    Parameters
    ----------
    results : list
        Iterable of episode result objects with attributes `steps`,
        `max_floor`, `outcome`, `elapsed_s`, `total_reward`.
    iteration : int
        Current training iteration (used only for quarantine filename).
    quarantine_dir : pathlib.Path, optional
        If provided and a quality check fails, the offending batch is
        dumped here as `qc_fail_iter<iteration>.json`.

    Returns
    -------
    (passed, reason) : tuple[bool, str]
    """
    valid = [r for r in results if r.steps > 0]
    if not valid:
        return False, "no valid episodes"

    avg_steps = np.mean([r.steps for r in valid])
    max_floor = max(r.max_floor for r in valid)
    budget_hits = sum(1 for r in valid if r.outcome == "truncated" and r.max_floor <= 2)
    budget_pct = budget_hits / len(valid)

    reasons = []
    if avg_steps < 10:
        reasons.append(f"avg_steps={avg_steps:.0f}<10")
    if max_floor < 2:
        reasons.append(f"max_floor={max_floor}<2")
    if budget_pct > 0.25:
        reasons.append(f"floor_budget_hits={budget_pct:.0%}>25%")
    nan_count = sum(1 for r in valid if np.isnan(r.total_reward))
    if nan_count > 0:
        reasons.append(f"nan_rewards={nan_count}")

    if reasons:
        reason = "; ".join(reasons)
        if quarantine_dir:
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            qc_file = quarantine_dir / f"qc_fail_iter{iteration:05d}.json"
            try:
                qc_file.write_text(json.dumps({
                    "iteration": iteration, "reason": reason,
                    "episodes": [{"steps": r.steps, "max_floor": r.max_floor,
                                  "outcome": r.outcome, "elapsed_s": r.elapsed_s,
                                  "total_reward": r.total_reward}
                                 for r in valid],
                }, indent=1, default=str))
            except Exception:
                pass
        return False, reason
    return True, "ok"


@dataclass
class Alert:
    iteration: int
    severity: str  # "warning" or "error"
    message: str

    def __str__(self):
        return f"[{self.severity.upper()}] iter {self.iteration}: {self.message}"


class TrainingHealthMonitor:
    """Mode-agnostic training health monitor."""

    def __init__(
        self,
        loss_fields: list[str] | None = None,
        entropy_field: str | None = None,
        floor_field: str = "avg_floor",
        loss_spike_threshold: float = 5.0,
        min_entropy: float = 0.1,
        max_entropy_ratio: float = 0.95,
        floor_stall_window: int = 50,
        floor_stall_min_improvement: float = 0.1,
        max_value_loss: float = 10.0,
    ):
        self.loss_fields = loss_fields
        self.entropy_field = entropy_field
        self.floor_field = floor_field
        self.loss_spike_threshold = loss_spike_threshold
        self.min_entropy = min_entropy
        self.max_entropy_ratio = max_entropy_ratio
        self.floor_stall_window = floor_stall_window
        self.floor_stall_min_improvement = floor_stall_min_improvement
        self.max_value_loss = max_value_loss

    def _auto_detect(self, entries: list[dict]) -> None:
        """Auto-detect field names from first entry if not configured."""
        if not entries:
            return
        keys = set(entries[0].keys())

        if self.loss_fields is None:
            self.loss_fields = []
            for f in ["policy_loss", "value_loss", "ppo_ploss", "ppo_vloss",
                       "mcts_ploss", "mcts_vloss"]:
                if f in keys:
                    self.loss_fields.append(f)

        if self.entropy_field is None:
            for f in ["entropy", "ppo_entropy"]:
                if f in keys:
                    self.entropy_field = f
                    break

    def check_loss_spike(self, entries: list[dict], field: str,
                         threshold: float | None = None) -> list[Alert]:
        """Flag iterations where loss > threshold * rolling median."""
        threshold = threshold or self.loss_spike_threshold
        alerts = []
        values = [e.get(field, 0) for e in entries]
        window = 20

        for i in range(window, len(values)):
            recent = values[max(0, i - window):i]
            recent_clean = [v for v in recent if _is_finite(v) and v != 0]
            if not recent_clean:
                continue
            median = sorted(recent_clean)[len(recent_clean) // 2]
            if median == 0:
                continue
            curr = values[i]
            if _is_finite(curr) and abs(curr) > threshold * abs(median):
                alerts.append(Alert(
                    entries[i]["iteration"], "warning",
                    f"{field} spike: {curr:.4f} > {threshold}x median({median:.4f})"
                ))
        return alerts

    def check_entropy_collapse(self, entries: list[dict],
                                field: str | None = None) -> list[Alert]:
        """Flag when entropy drops below minimum (policy collapse)."""
        field = field or self.entropy_field
        if not field:
            return []
        alerts = []
        for e in entries:
            val = e.get(field)
            if val is not None and _is_finite(val) and val < self.min_entropy and val != 0:
                alerts.append(Alert(
                    e["iteration"], "warning",
                    f"entropy collapse: {field}={val:.4f} < {self.min_entropy}"
                ))
        return alerts

    def check_floor_stall(self, entries: list[dict]) -> list[Alert]:
        """Flag when floor hasn't improved over a window of iterations."""
        alerts = []
        field = self.floor_field
        window = self.floor_stall_window
        min_imp = self.floor_stall_min_improvement

        if len(entries) < window:
            return []

        for i in range(window, len(entries)):
            start_floor = entries[i - window].get(field, 0)
            end_floor = entries[i].get(field, 0)
            if _is_finite(start_floor) and _is_finite(end_floor):
                improvement = end_floor - start_floor
                if improvement < min_imp and i == len(entries) - 1:
                    # Only alert on the latest iteration to avoid spam
                    alerts.append(Alert(
                        entries[i]["iteration"], "warning",
                        f"floor stall: {field} improved only {improvement:.2f} "
                        f"over last {window} iters ({start_floor:.1f} -> {end_floor:.1f})"
                    ))
        return alerts

    def check_nan_inf(self, entries: list[dict],
                       fields: list[str] | None = None) -> list[Alert]:
        """Scan numeric fields for NaN/Inf values."""
        alerts = []
        for e in entries:
            for k, v in e.items():
                if fields and k not in fields:
                    continue
                if isinstance(v, (int, float)) and not _is_finite(v):
                    alerts.append(Alert(
                        e["iteration"], "error",
                        f"NaN/Inf detected: {k}={v}"
                    ))
        return alerts

    def check_value_divergence(self, entries: list[dict]) -> list[Alert]:
        """Flag when value loss exceeds threshold (diverging value head)."""
        alerts = []
        vloss_fields = [f for f in (self.loss_fields or [])
                        if "value" in f or "vloss" in f]
        for field in vloss_fields:
            for e in entries:
                val = e.get(field)
                if val is not None and _is_finite(val) and abs(val) > self.max_value_loss:
                    alerts.append(Alert(
                        e["iteration"], "warning",
                        f"value divergence: {field}={val:.4f} > {self.max_value_loss}"
                    ))
        return alerts

    def check_all(self, entries: list[dict]) -> list[Alert]:
        """Run all health checks. Returns sorted alert list."""
        if not entries:
            return []

        self._auto_detect(entries)
        alerts = []

        # Loss spikes
        for field in (self.loss_fields or []):
            alerts.extend(self.check_loss_spike(entries, field))

        # Entropy
        alerts.extend(self.check_entropy_collapse(entries))

        # Floor stall
        alerts.extend(self.check_floor_stall(entries))

        # NaN/Inf
        alerts.extend(self.check_nan_inf(entries))

        # Value divergence
        alerts.extend(self.check_value_divergence(entries))

        alerts.sort(key=lambda a: a.iteration)
        return alerts

    def summary(self, entries: list[dict]) -> str:
        """Generate a human-readable health summary."""
        if not entries:
            return "No data."

        self._auto_detect(entries)
        alerts = self.check_all(entries)
        n = len(entries)
        last = entries[-1]

        lines = [f"Training Health Report ({n} iterations)"]
        lines.append("=" * 50)

        # Key metrics
        if self.floor_field in last:
            lines.append(f"  Floor: {last[self.floor_field]:.1f}")
        if "victories" in last:
            lines.append(f"  Victories (last): {last['victories']}")
        for f in (self.loss_fields or []):
            if f in last:
                lines.append(f"  {f}: {last[f]:.4f}")
        if self.entropy_field and self.entropy_field in last:
            lines.append(f"  {self.entropy_field}: {last[self.entropy_field]:.4f}")

        # Alerts
        if alerts:
            errors = [a for a in alerts if a.severity == "error"]
            warnings = [a for a in alerts if a.severity == "warning"]
            lines.append(f"\nAlerts: {len(errors)} errors, {len(warnings)} warnings")
            for a in alerts[:20]:  # cap at 20
                lines.append(f"  {a}")
            if len(alerts) > 20:
                lines.append(f"  ... and {len(alerts) - 20} more")
        else:
            lines.append("\nNo alerts — training looks healthy.")

        return "\n".join(lines)


def _is_finite(v) -> bool:
    """Check if value is finite (not NaN, not Inf)."""
    if isinstance(v, float):
        return math.isfinite(v)
    if isinstance(v, int):
        return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Training health analysis")
    parser.add_argument("metrics_file", help="Path to metrics.jsonl")
    parser.add_argument("--loss-threshold", type=float, default=5.0)
    parser.add_argument("--min-entropy", type=float, default=0.1)
    parser.add_argument("--stall-window", type=int, default=50)
    args = parser.parse_args()

    path = Path(args.metrics_file)
    if not path.exists():
        # Try glob
        import glob
        matches = glob.glob(args.metrics_file)
        if matches:
            path = Path(max(matches, key=lambda p: Path(p).stat().st_mtime))
        else:
            print(f"File not found: {args.metrics_file}")
            sys.exit(1)

    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    monitor = TrainingHealthMonitor(
        loss_spike_threshold=args.loss_threshold,
        min_entropy=args.min_entropy,
        floor_stall_window=args.stall_window,
    )
    print(monitor.summary(entries))
    print(f"\nSource: {path}")


if __name__ == "__main__":
    main()
