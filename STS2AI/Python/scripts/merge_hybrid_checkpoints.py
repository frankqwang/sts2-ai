#!/usr/bin/env python3
"""Merge separately trained PPO and combat checkpoints into one hybrid checkpoint.

Default ownership:
- PPO checkpoint owns ``ppo_model`` and shared weights.
- Combat checkpoint owns combat-specific ``combat_model`` weights.
- Shared combat keys such as ``entity_emb.*`` and ``symbolic_head.*`` are
  copied from PPO into the combat state when shapes match, so runtime loading
  does not accidentally overwrite the PPO-side shared modules with stale combat
  copies.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    python_root = Path(__file__).resolve().parents[1]
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

import _path_init  # noqa: F401

import torch

from checkpoint_compat import (
    COMBAT_MODEL_KEY,
    get_combat_model_config,
    get_combat_model_state,
    make_hybrid_checkpoint_payload,
)
from sts2ai_paths import ARTIFACTS_ROOT


SHARED_COMBAT_KEY_PREFIXES = (
    "entity_emb.",
    "symbolic_head.",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dict: {path}")
    return checkpoint


def _get_ppo_state(checkpoint: dict[str, Any], path: Path) -> dict[str, Any]:
    state = checkpoint.get("ppo_model") or checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"PPO checkpoint has no ppo_model/model_state_dict: {path}")
    return state


def _infer_combat_model_config(state: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(fallback or {})
    card_weight = state.get("entity_emb.card_embed.weight")
    action_proj = state.get("action_proj.weight")
    if isinstance(card_weight, torch.Tensor) and card_weight.ndim == 2:
        config.setdefault("embed_dim", int(card_weight.shape[1]))
    if isinstance(action_proj, torch.Tensor) and action_proj.ndim == 2:
        config.setdefault("hidden_dim", int(action_proj.shape[0]))
    if "combat_main_path_mode" not in config:
        config["combat_main_path_mode"] = "light_attention" if any("deck_encoder" in key for key in state) else "mlp"
    return config


def _tensor_shape(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    return None


def _same_shape(left: Any, right: Any) -> bool:
    return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and left.shape == right.shape


def _copy_shared_weights_from_ppo(
    *,
    ppo_state: dict[str, Any],
    combat_state: dict[str, Any],
    shared_prefixes: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    merged = dict(combat_state)
    aligned: list[str] = []
    missing_in_ppo: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []

    for key, combat_value in combat_state.items():
        if not key.startswith(shared_prefixes):
            continue
        ppo_value = ppo_state.get(key)
        if ppo_value is None:
            missing_in_ppo.append(key)
            continue
        if _same_shape(ppo_value, combat_value):
            merged[key] = ppo_value
            aligned.append(key)
            continue
        shape_mismatches.append(
            {
                "key": key,
                "ppo_shape": _tensor_shape(ppo_value),
                "combat_shape": _tensor_shape(combat_value),
            }
        )

    report = {
        "shared_prefixes": list(shared_prefixes),
        "aligned_count": len(aligned),
        "aligned_keys_preview": aligned[:25],
        "missing_in_ppo_count": len(missing_in_ppo),
        "missing_in_ppo_preview": missing_in_ppo[:25],
        "shape_mismatch_count": len(shape_mismatches),
        "shape_mismatches_preview": shape_mismatches[:25],
    }
    return merged, report


def _state_dict_report(name: str, state: dict[str, Any]) -> dict[str, Any]:
    tensor_items = [(key, value) for key, value in state.items() if isinstance(value, torch.Tensor)]
    total_params = int(sum(value.numel() for _key, value in tensor_items))
    return {
        "name": name,
        "key_count": len(state),
        "tensor_key_count": len(tensor_items),
        "total_params": total_params,
        "keys_preview": sorted(state.keys())[:25],
    }


def _compare_shared_against_ppo(
    *,
    ppo_state: dict[str, Any],
    combat_state: dict[str, Any],
    shared_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    shared_keys = sorted(key for key in combat_state if key.startswith(shared_prefixes))
    identical = []
    different = []
    missing = []
    for key in shared_keys:
        ppo_value = ppo_state.get(key)
        combat_value = combat_state.get(key)
        if ppo_value is None:
            missing.append(key)
            continue
        if _same_shape(ppo_value, combat_value) and torch.equal(ppo_value, combat_value):
            identical.append(key)
        else:
            different.append(
                {
                    "key": key,
                    "ppo_shape": _tensor_shape(ppo_value),
                    "combat_shape": _tensor_shape(combat_value),
                    "same_shape": _same_shape(ppo_value, combat_value),
                }
            )
    return {
        "shared_key_count": len(shared_keys),
        "identical_count": len(identical),
        "different_count": len(different),
        "missing_in_ppo_count": len(missing),
        "different_preview": different[:25],
        "missing_in_ppo_preview": missing[:25],
    }


def _default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ARTIFACTS_ROOT / "checkpoint_merges" / f"hybrid_merged_{stamp}.pt"


def _write_report(report: dict[str, Any], output_path: Path) -> tuple[Path, Path]:
    json_path = output_path.with_suffix(".merge_report.json")
    md_path = output_path.with_suffix(".merge_report.md")
    json_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Hybrid Checkpoint Merge Report",
        "",
        f"- 输出 checkpoint: `{output_path}`",
        f"- PPO 来源: `{report['sources']['ppo_checkpoint']}`",
        f"- Combat 来源: `{report['sources']['combat_checkpoint']}`",
        f"- Base 来源: `{report['sources'].get('base_checkpoint') or '<none>'}`",
        f"- 合并时间: `{report['created_at']}`",
        "",
        "## 权重摘要",
        "",
        f"- PPO keys: `{report['ppo_state']['key_count']}`，参数量: `{report['ppo_state']['total_params']}`",
        f"- Combat keys: `{report['combat_state']['key_count']}`，参数量: `{report['combat_state']['total_params']}`",
        f"- 共享 key 对齐数量: `{report['shared_alignment']['aligned_count']}`",
        f"- 共享 key shape mismatch: `{report['shared_alignment']['shape_mismatch_count']}`",
        "",
        "## 配置",
        "",
        f"- `ppo_config`: `{json.dumps(_json_safe(report['ppo_config']), ensure_ascii=False)}`",
        f"- `combat_model_config`: `{json.dumps(_json_safe(report['combat_model_config']), ensure_ascii=False)}`",
        "",
        "## 结论",
        "",
        f"- status: `{report['status']}`",
    ]
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def merge_checkpoints(args: argparse.Namespace) -> dict[str, Any]:
    ppo_path = Path(args.ppo_checkpoint).resolve()
    combat_path = Path(args.combat_checkpoint).resolve()
    base_path = Path(args.base_checkpoint).resolve() if args.base_checkpoint else None
    output_path = Path(args.output).resolve() if args.output else _default_output_path().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ppo_ckpt = _load_checkpoint(ppo_path)
    combat_ckpt = _load_checkpoint(combat_path)
    base_ckpt = _load_checkpoint(base_path) if base_path else {}

    ppo_state = _get_ppo_state(ppo_ckpt, ppo_path)
    raw_combat_state = get_combat_model_state(combat_ckpt, allow_standalone=True)
    if not isinstance(raw_combat_state, dict):
        raise ValueError(f"Combat checkpoint has no combat_model/model_state_dict: {combat_path}")

    shared_prefixes = tuple(args.shared_prefix or SHARED_COMBAT_KEY_PREFIXES)
    pre_alignment = _compare_shared_against_ppo(
        ppo_state=ppo_state,
        combat_state=raw_combat_state,
        shared_prefixes=shared_prefixes,
    )
    combat_state, shared_alignment = _copy_shared_weights_from_ppo(
        ppo_state=ppo_state,
        combat_state=raw_combat_state,
        shared_prefixes=shared_prefixes,
    )

    if shared_alignment["shape_mismatch_count"] > 0 and args.strict_shared:
        raise ValueError(
            "Shared PPO/combat keys have shape mismatches. "
            "Re-run with --no-strict-shared only if you intentionally accept this."
        )

    ppo_config = dict(base_ckpt.get("ppo_config") or ppo_ckpt.get("ppo_config") or {})
    combat_config_source = (
        get_combat_model_config(combat_ckpt)
        or get_combat_model_config(base_ckpt)
        or {}
    )
    combat_model_config = _infer_combat_model_config(raw_combat_state, combat_config_source)

    iterations = {
        "ppo": ppo_ckpt.get("iteration"),
        "combat": combat_ckpt.get("iteration"),
        "base": base_ckpt.get("iteration") if base_ckpt else None,
    }
    numeric_iterations = [int(value) for value in iterations.values() if isinstance(value, int)]
    merged_iteration = max(numeric_iterations) if numeric_iterations else 0

    warnings: list[str] = []
    if pre_alignment["different_count"] > 0:
        warnings.append(
            "Combat checkpoint had shared keys that differed from PPO; merged checkpoint uses PPO-owned shared weights."
        )
    if shared_alignment["missing_in_ppo_count"] > 0:
        warnings.append("Some combat shared keys were missing from PPO and were left unchanged.")

    merged_metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ppo_checkpoint": str(ppo_path),
        "combat_checkpoint": str(combat_path),
        "base_checkpoint": str(base_path) if base_path else None,
        "source_iterations": iterations,
        "shared_weight_owner": "ppo_model",
        "shared_prefixes": list(shared_prefixes),
    }

    payload = make_hybrid_checkpoint_payload(
        ppo_model=ppo_state,
        combat_model=combat_state,
        ppo_config=ppo_config,
        combat_model_config=combat_model_config,
        iteration=merged_iteration,
        merged_checkpoint=merged_metadata,
    )

    if args.include_extra_metadata and base_ckpt:
        for key in ("vocab", "vocab_hash", "run_config", "metadata"):
            if key in base_ckpt and key not in payload:
                payload[key] = base_ckpt[key]

    torch.save(payload, output_path)

    report = {
        "status": "ok",
        "created_at": merged_metadata["created_at"],
        "output_checkpoint": str(output_path),
        "sources": merged_metadata,
        "iteration": merged_iteration,
        "ppo_config": ppo_config,
        "combat_model_config": combat_model_config,
        "ppo_state": _state_dict_report("ppo_model", ppo_state),
        "combat_state": _state_dict_report(COMBAT_MODEL_KEY, combat_state),
        "shared_before_alignment": pre_alignment,
        "shared_alignment": shared_alignment,
        "warnings": warnings,
    }
    json_path, md_path = _write_report(report, output_path)
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge PPO and combat checkpoints into a canonical hybrid checkpoint."
    )
    parser.add_argument("--ppo-checkpoint", required=True, help="Checkpoint that provides ppo_model.")
    parser.add_argument("--combat-checkpoint", required=True, help="Checkpoint that provides combat_model/model_state_dict.")
    parser.add_argument("--base-checkpoint", default=None, help="Optional metadata/config source.")
    parser.add_argument("--output", default=None, help="Output .pt path. Defaults to STS2AI/Artifacts/checkpoint_merges/.")
    parser.add_argument(
        "--shared-prefix",
        action="append",
        default=None,
        help="Shared state_dict key prefix owned by PPO. Can be repeated. Defaults to entity_emb. and symbolic_head.",
    )
    parser.add_argument(
        "--no-strict-shared",
        dest="strict_shared",
        action="store_false",
        help="Do not fail on shared-key shape mismatches.",
    )
    parser.add_argument(
        "--include-extra-metadata",
        action="store_true",
        default=False,
        help="Copy selected non-weight metadata from --base-checkpoint when present.",
    )
    parser.set_defaults(strict_shared=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = merge_checkpoints(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Merged checkpoint: {report['output_checkpoint']}")
    print(f"Report JSON: {report['report_json']}")
    print(f"Report MD: {report['report_md']}")
    if report.get("warnings"):
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
