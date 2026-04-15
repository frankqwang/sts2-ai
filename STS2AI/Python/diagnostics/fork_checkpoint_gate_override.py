"""Fork a hybrid checkpoint and override specific gate scalars in combat_model.

Why: `action_aux_gate` (and its 7 siblings) in CombatPolicyValueNetwork starts
at 0.1 / 0.0 and barely moves in a few training iters, so the lethal-aware
per-action features (kills_target, prevented_lethal, ...) only contribute ~10%
to the action representation. Simply editing the nn.Parameter init does NOT
help when we resume from an existing checkpoint — the old gate value loads
back. This script lets us fork a checkpoint and stamp the gate to any value,
then resume training from the forked file.

Usage:
  python STS2AI/Python/diagnostics/fork_checkpoint_gate_override.py \
     --input  STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175.pt \
     --output STS2AI/Assets/checkpoints/act1/retrieval_final_iter2175_gate10.pt \
     --set action_aux_gate=1.0

Any `--set key=val` may be repeated. Supports both scalar (single-element)
and vector gates. Unknown keys cause an error unless --allow-missing is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_assignments(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"bad --set (no '='): {item}")
        key, val = item.split("=", 1)
        out[key.strip()] = float(val.strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source checkpoint path")
    parser.add_argument("--output", required=True, help="Destination checkpoint path")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="key=value assignment (repeatable). key is the trailing name of the "
             "gate, e.g. 'action_aux_gate' or 'main_action_context_gate'.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Silently skip keys that don't exist in combat_model.",
    )
    args = parser.parse_args()

    assignments = parse_assignments(args.set)
    if not assignments:
        raise SystemExit("pass at least one --set key=value")

    src = Path(args.input)
    dst = Path(args.output)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    combat_state = ckpt.get("combat_model") or ckpt.get("model_state_dict")
    if not isinstance(combat_state, dict):
        raise SystemExit(f"No combat_model state_dict found in {src}")

    # Build lookup of available gate-like keys (scalar params). Keys in the
    # state_dict are already the final names ("action_aux_gate",
    # "main_action_context_gate", etc).
    available = {k: v for k, v in combat_state.items() if isinstance(v, torch.Tensor)}

    summary: list[str] = []
    for key, val in assignments.items():
        if key not in available:
            if args.allow_missing:
                summary.append(f"  SKIP  {key} (not in checkpoint)")
                continue
            raise SystemExit(f"key '{key}' not found in combat_model. Keys available: "
                             f"{[k for k in available if 'gate' in k.lower()]}")
        old = available[key]
        if old.numel() == 1:
            new = torch.tensor(val, dtype=old.dtype)
        else:
            new = torch.full_like(old, val)
        combat_state[key] = new
        old_v = old.item() if old.numel() == 1 else old.tolist()
        new_v = new.item() if new.numel() == 1 else new.tolist()
        summary.append(f"  {key:<50}: {old_v}  →  {new_v}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    print(f"Wrote {dst}")
    for line in summary:
        print(line)


if __name__ == "__main__":
    main()
