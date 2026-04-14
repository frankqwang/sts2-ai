#!/usr/bin/env python3
"""Export runtime card descriptions and upgrade previews via HeadlessSim."""

from __future__ import annotations

import argparse
import subprocess
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HEADLESS_SIM_CSPROJ = REPO_ROOT / "STS2AI" / "ENV" / "Sim" / "Host" / "headless_sim_host_0991.csproj"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "raw" / "card_runtime_texts.json"
DEFAULT_BUILD_OUTPUT_DIR = REPO_ROOT / "STS2AI" / "ENV" / "Sim" / "Host" / "bin" / "runtime_card_text_export"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export runtime card texts from HeadlessSim.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--locales", default="eng,zhs", help="逗号分隔的语言列表，默认 eng,zhs")
    parser.add_argument("--build-output-dir", type=Path, default=DEFAULT_BUILD_OUTPUT_DIR)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.build_output_dir.mkdir(parents=True, exist_ok=True)

    build_command = [
        "dotnet",
        "build",
        str(HEADLESS_SIM_CSPROJ),
        "-c",
        "Debug",
        "-o",
        str(args.build_output_dir),
    ]
    subprocess.run(build_command, check=True, cwd=REPO_ROOT)

    localization_src = REPO_ROOT / "localization"
    localization_dst = args.build_output_dir / "localization"
    if localization_dst.exists():
        shutil.rmtree(localization_dst)
    shutil.copytree(localization_src, localization_dst)

    dll_path = args.build_output_dir / "headless_sim_host_0991.dll"
    command = [
        "dotnet",
        str(dll_path),
        "--export-card-runtime-texts",
        str(args.output),
        "--export-locales",
        args.locales,
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
