"""Write metrics.json for LLM dataset, SFT, and spectate runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.metrics import (  # noqa: E402
    summarize_dataset_dir,
    summarize_sft_run,
    summarize_spectate_run,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="")
    parser.add_argument("--sft-run-dir", type=str, default="")
    parser.add_argument("--spectate-run-dir", type=str, default="")
    parser.add_argument("--trace", type=str, default="")
    parser.add_argument("--spectate-stdout", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict
    default_out: Path

    if args.sft_run_dir:
        run_root = Path(args.sft_run_dir)
        dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
        payload = summarize_sft_run(run_root, dataset_dir=dataset_dir)
        default_out = run_root / "metrics.json"
    elif args.spectate_run_dir or args.trace or args.spectate_stdout:
        run_root = Path(args.spectate_run_dir) if args.spectate_run_dir else Path(args.trace or args.spectate_stdout).parent
        trace = Path(args.trace) if args.trace else None
        stdout = Path(args.spectate_stdout) if args.spectate_stdout else None
        manifest = Path(args.manifest) if args.manifest else None
        payload = summarize_spectate_run(run_root, trace_path=trace, stdout_path=stdout, manifest_path=manifest)
        default_out = run_root / "metrics.json"
    elif args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        payload = {"kind": "dataset", **summarize_dataset_dir(dataset_dir)}
        default_out = dataset_dir / "metrics.json"
    else:
        raise SystemExit("provide --dataset-dir, --sft-run-dir, or --spectate-run-dir/--trace")

    out = Path(args.out) if args.out else default_out
    write_json(out, payload)
    print(f"[metrics] wrote {out}")


if __name__ == "__main__":
    main()
