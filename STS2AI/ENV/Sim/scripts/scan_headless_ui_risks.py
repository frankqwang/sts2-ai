#!/usr/bin/env python3
"""基于 headless host 实际编译边界扫描 UI / 表现层空引用风险。"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
HOST_PROJECT = REPO_ROOT / "STS2AI" / "ENV" / "Sim" / "HeadlessSim" / "HeadlessSim.csproj"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "STS2AI" / "Artifacts" / "headless_ui_risk_scan"


@dataclass(frozen=True)
class PatternSpec:
    pattern_id: str
    priority: int
    title: str
    regex: re.Pattern[str]
    rationale: str


@dataclass
class Finding:
    file: str
    line: int
    priority: int
    pattern_id: str
    title: str
    rationale: str
    source_kind: str
    directory_bucket: str
    mitigation: str
    member_hint: str | None
    code: str


LINE_PATTERNS: tuple[PatternSpec, ...] = (
    PatternSpec(
        pattern_id="direct_room_background",
        priority=1,
        title="直接访问 NCombatRoom 背景节点",
        regex=re.compile(r"\bNCombatRoom\.Instance\.Background\b"),
        rationale="headless / pure-sim 下战斗房间背景节点可能不存在，直接走 Background 往往是表现层假设。",
    ),
    PatternSpec(
        pattern_id="direct_room_creature_node",
        priority=1,
        title="直接访问 NCombatRoom.GetCreatureNode",
        regex=re.compile(r"\bNCombatRoom\.Instance\.GetCreatureNode\("),
        rationale="headless 下 NCombatRoom.Instance 可能为空；直接取 CreatureNode 的怪物/特效逻辑是高频 NPE 来源。",
    ),
    PatternSpec(
        pattern_id="direct_room_vfx_container",
        priority=1,
        title="直接访问 NCombatRoom VFX 容器",
        regex=re.compile(r"\bNCombatRoom\.Instance\.(?:Back)?CombatVfxContainer\b"),
        rationale="CombatVfxContainer 属于场景树表现层对象；pure-sim 里直接依赖它容易空引用。",
    ),
    PatternSpec(
        pattern_id="direct_room_visual_effect",
        priority=1,
        title="直接触发 NCombatRoom 表现层效果",
        regex=re.compile(r"\bNCombatRoom\.Instance\.(?:RadialBlur|PlaySplashVfx|ShakeOstyIfDead|RemoveCreatureNode|SetCreatureIsInteractable)\("),
        rationale="这些接口都依赖房间场景树或 creature node，pure-sim 下通常应该做 best-effort/no-op。",
    ),
    PatternSpec(
        pattern_id="direct_game_feedback",
        priority=1,
        title="直接访问 NGame 视觉/震屏反馈",
        regex=re.compile(r"\bNGame\.Instance\.(?:ScreenShake|ScreenShakeTrauma|ScreenRumble|DoHitStop|CurrentRunNode|GetViewportRect)\b"),
        rationale="这些调用依赖全局 UI / viewport；headless 环境里通常只能 no-op 或走空判断。",
    ),
    PatternSpec(
        pattern_id="direct_special_node_access",
        priority=1,
        title="直接走 special node / spine visuals",
        regex=re.compile(r"(?<!\?)\.(?:GetSpecialNode<|SpineAnimation\.|SpineController\.)"),
        rationale="special node / spine controller 依赖具体 visual scene；headless 下要么判空，要么迁到 compat overlay。",
    ),
)

ASSIGNMENT_RE = re.compile(
    r"\b(?:var|[A-Za-z_][A-Za-z0-9_<>,.?]*)\s+(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<expr>.+?);"
)
SCENE_NULLABLE_EXPR_HINTS = (
    "NCombatRoom.Instance?",
    "NGame.Instance?",
    "GetNodeOrNull<",
    "GetSpecialNode<",
    "SpineController",
    "SpineAnimation",
)
MAYBE_NULL_GUARD_TEMPLATES = (
    r"\b{var}\s*!=\s*null\b",
    r"\b{var}\s+is\s+not\s+null\b",
    r"ThrowIfNull\(\s*{var}\s*\)",
)
DIRECT_DEREF_TEMPLATE = r"\b{var}\.(?!\?)"
MEMBER_HINT_RE = re.compile(
    r"^\s*(?:public|private|protected|internal)\s+(?:static\s+)?(?:async\s+)?(?:override\s+)?(?:virtual\s+)?(?:sealed\s+)?(?:partial\s+)?(?:unsafe\s+)?[A-Za-z_][A-Za-z0-9_<>,.? \[\]]*\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
)
PROPERTY_HINT_RE = re.compile(
    r"^\s*(?:public|private|protected|internal)\s+(?:static\s+)?(?:override\s+)?(?:virtual\s+)?(?:sealed\s+)?[A-Za-z_][A-Za-z0-9_<>,.? \[\]]*\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$"
)


def _expand_msbuild_value(raw_value: str, props: dict[str, str], *, host_project: Path) -> str | None:
    value = raw_value.strip()
    if not value:
        return None
    if "MSBuildThisFileDirectory" not in props:
        props["MSBuildThisFileDirectory"] = str(host_project.parent.resolve()) + os.sep
    for _ in range(8):
        matches = re.findall(r"\$\(([^)]+)\)", value)
        if not matches:
            break
        changed = False
        for name in matches:
            replacement = props.get(name)
            if replacement is None:
                return None
            value = value.replace(f"$({name})", replacement)
            changed = True
        if not changed:
            break
    if "$(" in value:
        return None
    return value


def _resolve_csproj_pattern(raw_value: str, *, props: dict[str, str], host_project: Path) -> str | None:
    expanded = _expand_msbuild_value(raw_value, props, host_project=host_project)
    if expanded is None:
        return None
    expanded = os.path.expandvars(expanded)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(str((host_project.parent / expanded).resolve()))


def _resolve_props(
    project_path: Path,
    *,
    root: ET.Element | None = None,
    inherited_props: dict[str, str] | None = None,
) -> dict[str, str]:
    if root is None:
        root = ET.parse(project_path).getroot()
    props: dict[str, str] = dict(inherited_props or {})
    props["MSBuildThisFileDirectory"] = str(project_path.parent.resolve()) + os.sep
    for prop_group in root.findall("PropertyGroup"):
        for child in prop_group:
            if child.text is None or not child.text.strip():
                continue
            expanded = _expand_msbuild_value(child.text, props, host_project=project_path)
            if expanded is not None:
                props[child.tag] = expanded
    props.setdefault("UpstreamRoot", str(REPO_ROOT.resolve()))
    return props


def _apply_compile_items(
    project_path: Path,
    *,
    compiled: dict[str, Path],
    inherited_props: dict[str, str] | None = None,
    seen: set[Path] | None = None,
) -> None:
    resolved_project = project_path.resolve()
    if seen is None:
        seen = set()
    if resolved_project in seen:
        return
    seen.add(resolved_project)

    root = ET.parse(resolved_project).getroot()
    props = _resolve_props(resolved_project, root=root, inherited_props=inherited_props)
    for child in list(root):
        if child.tag == "ItemGroup":
            for compile_node in child.findall("Compile"):
                include_raw = compile_node.get("Include")
                remove_raw = compile_node.get("Remove")
                if include_raw:
                    pattern = _resolve_csproj_pattern(include_raw, props=props, host_project=resolved_project)
                    if pattern is not None:
                        for match in glob.glob(pattern, recursive=True):
                            path = Path(match).resolve()
                            if path.is_file() and path.suffix.lower() == ".cs":
                                compiled[str(path)] = path
                if remove_raw:
                    pattern = _resolve_csproj_pattern(remove_raw, props=props, host_project=resolved_project)
                    if pattern is not None:
                        for match in glob.glob(pattern, recursive=True):
                            compiled.pop(str(Path(match).resolve()), None)
            continue

        if child.tag != "Import":
            continue

        import_raw = child.get("Project")
        if not import_raw:
            continue
        import_path = _resolve_csproj_pattern(import_raw, props=props, host_project=resolved_project)
        if import_path is None:
            continue
        import_file = Path(import_path)
        if import_file.is_file():
            _apply_compile_items(
                import_file,
                compiled=compiled,
                inherited_props=props,
                seen=seen,
            )


def _expand_compile_nodes(host_project: Path) -> list[Path]:
    compiled: dict[str, Path] = {}
    _apply_compile_items(host_project, compiled=compiled)
    return sorted(compiled.values())


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _source_kind(path: Path) -> str:
    rel = _safe_relative(path, REPO_ROOT)
    if rel.startswith("src/"):
        return "upstream_src"
    if rel.startswith("STS2AI/ENV/Sim/SrcCompat/AutoGenerated/"):
        return "autogenerated_src_compat"
    if rel.startswith("STS2AI/ENV/Sim/SrcCompat/"):
        return "src_compat"
    if rel.startswith("STS2AI/ENV/Sim/HeadlessSim/Simulation/"):
        return "overlay_simulation"
    if rel.startswith("STS2AI/ENV/Sim/HeadlessSim/Training/"):
        return "overlay_training"
    if rel.startswith("STS2AI/ENV/Sim/HeadlessSim/TestSupport/"):
        return "overlay_testsupport"
    if rel.startswith("STS2AI/ENV/Sim/HeadlessSim/"):
        return "runtime_headless"
    return "other"


def _mitigation(path: Path) -> str:
    rel = _safe_relative(path, REPO_ROOT)
    compat_prefix = "STS2AI/ENV/Sim/SrcCompat/Source01032/Core/Models/"
    autogenerated_prefix = "STS2AI/ENV/Sim/SrcCompat/AutoGenerated/Core/Models/"
    if rel.startswith("src/Core/Models/"):
        suffix = rel.removeprefix("src/Core/Models/")
        compat_peer = REPO_ROOT / compat_prefix / suffix
        autogenerated_peer = REPO_ROOT / autogenerated_prefix / suffix
        return "review_existing_src_compat" if compat_peer.exists() or autogenerated_peer.exists() else "add_src_compat"
    if rel.startswith(compat_prefix):
        return "tighten_existing_src_compat"
    if rel.startswith(autogenerated_prefix):
        return "improve_generated_src_compat"
    return "manual_review"


def _directory_bucket(path: Path) -> str:
    rel = _safe_relative(path, REPO_ROOT)
    parts = rel.split("/")
    if len(parts) >= 4 and parts[0] == "src" and parts[1] == "Core":
        if parts[2] == "Models" and len(parts) >= 5:
            return f"models/{parts[3].lower()}"
        return f"core/{parts[2].lower()}"
    if rel.startswith("STS2AI/ENV/Sim/SrcCompat/AutoGenerated/Core/Models/"):
        sub = parts[8].lower() if len(parts) >= 10 else "models"
        return f"src_compat/autogenerated/{sub}"
    if rel.startswith("STS2AI/ENV/Sim/SrcCompat/Source01032/Core/Models/"):
        sub = parts[8].lower() if len(parts) >= 10 else "models"
        return f"src_compat/manual/{sub}"
    if rel.startswith("STS2AI/ENV/Sim/HeadlessSim/Simulation/"):
        return "overlay_simulation"
    if rel.startswith("STS2AI/ENV/Sim/HeadlessSim/Training/"):
        return "overlay_training"
    if rel.startswith("STS2AI/ENV/Sim/HeadlessSim/TestSupport/"):
        return "overlay_testsupport"
    if rel.startswith("STS2AI/ENV/Sim/HeadlessSim/"):
        return "runtime_headless"
    return "other"


def _member_hint(lines: list[str], index: int) -> str | None:
    start = max(0, index - 20)
    for candidate in range(index, start - 1, -1):
        line = lines[candidate].rstrip()
        match = MEMBER_HINT_RE.match(line)
        if match:
            return match.group("name")
        match = PROPERTY_HINT_RE.match(line)
        if match and candidate + 1 < len(lines) and lines[candidate + 1].strip().startswith("{"):
            return match.group("name")
    return None


def _in_scope(path: Path, scope: str) -> bool:
    rel = _safe_relative(path, REPO_ROOT)
    if scope == "compiled-all":
        return True
    if scope == "compiled-models":
        return (
            rel.startswith("src/Core/Models/")
            or rel.startswith("STS2AI/ENV/Sim/SrcCompat/Source01032/Core/Models/")
            or rel.startswith("STS2AI/ENV/Sim/SrcCompat/AutoGenerated/Core/Models/")
        )
    if scope == "compiled-monsters":
        return (
            rel.startswith("src/Core/Models/Monsters/")
            or rel.startswith("STS2AI/ENV/Sim/SrcCompat/Source01032/Core/Models/Monsters/")
            or rel.startswith("STS2AI/ENV/Sim/SrcCompat/AutoGenerated/Core/Models/Monsters/")
        )
    raise ValueError(f"unsupported scope: {scope}")


def _recently_guarded(lines: list[str], index: int, var_name: str) -> bool:
    patterns = [re.compile(template.format(var=re.escape(var_name))) for template in MAYBE_NULL_GUARD_TEMPLATES]
    for candidate in range(max(0, index - 2), index + 1):
        line = lines[candidate]
        if any(pattern.search(line) for pattern in patterns):
            return True
    return False


def _scan_file(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    findings: list[Finding] = []
    active_nullable_vars: dict[str, int] = {}
    brace_depth = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("//"):
            brace_depth += line.count("{") - line.count("}")
            continue
        direct_hit = next((pattern for pattern in LINE_PATTERNS if pattern.regex.search(line)), None)
        if (
            direct_hit is not None
            and direct_hit.pattern_id == "direct_special_node_access"
            and stripped.startswith(".GetSpecialNode<")
            and index > 0
            and "?." in lines[index - 1]
        ):
            direct_hit = None
        if direct_hit is not None:
            findings.append(
                Finding(
                    file=str(path.resolve()),
                    line=index + 1,
                    priority=direct_hit.priority,
                    pattern_id=direct_hit.pattern_id,
                    title=direct_hit.title,
                    rationale=direct_hit.rationale,
                    source_kind=_source_kind(path),
                    directory_bucket=_directory_bucket(path),
                    mitigation=_mitigation(path),
                    member_hint=_member_hint(lines, index),
                    code=stripped,
                )
            )
        else:
            for var_name, _ in list(active_nullable_vars.items()):
                deref_re = re.compile(DIRECT_DEREF_TEMPLATE.format(var=re.escape(var_name)))
                if deref_re.search(line) and not _recently_guarded(lines, index, var_name):
                    findings.append(
                        Finding(
                            file=str(path.resolve()),
                            line=index + 1,
                            priority=2,
                            pattern_id="nullable_var_dereference",
                            title="可能为空的局部变量被直接解引用",
                            rationale="变量来源已经带 scene/UI 可空信号，后续继续直接点号访问时很容易在 headless 场景里漏判空。",
                            source_kind=_source_kind(path),
                            directory_bucket=_directory_bucket(path),
                            mitigation=_mitigation(path),
                            member_hint=_member_hint(lines, index),
                            code=stripped,
                        )
                    )
                    break
        assignment = ASSIGNMENT_RE.search(line)
        if assignment:
            expr = assignment.group("expr")
            if any(hint in expr for hint in SCENE_NULLABLE_EXPR_HINTS):
                active_nullable_vars[assignment.group("var")] = brace_depth
        closing_depth = brace_depth + line.count("{") - line.count("}")
        expired = [name for name, depth in active_nullable_vars.items() if depth > closing_depth]
        for name in expired:
            active_nullable_vars.pop(name, None)
        brace_depth = closing_depth
    return findings


def _build_report(files: Iterable[Path], findings: list[Finding]) -> dict[str, object]:
    files = list(files)
    by_priority = Counter(f"P{finding.priority}" for finding in findings)
    by_pattern = Counter(finding.pattern_id for finding in findings)
    by_source_kind = Counter(finding.source_kind for finding in findings)
    by_bucket = Counter(finding.directory_bucket for finding in findings)
    by_mitigation = Counter(finding.mitigation for finding in findings)
    top_files = Counter(finding.file for finding in findings).most_common(40)
    candidate_files = []
    for file, count in top_files:
        file_findings = [finding for finding in findings if finding.file == file]
        mitigations = Counter(finding.mitigation for finding in file_findings)
        candidate_files.append(
            {
                "file": file,
                "count": count,
                "mitigation": mitigations.most_common(1)[0][0],
                "patterns": dict(Counter(finding.pattern_id for finding in file_findings)),
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT.resolve()),
        "host_project": str(HOST_PROJECT.resolve()),
        "effective_compile_file_count": len(files),
        "finding_count": len(findings),
        "summary": {
            "by_priority": dict(by_priority),
            "by_pattern": dict(by_pattern),
            "by_source_kind": dict(by_source_kind),
            "by_mitigation": dict(by_mitigation),
            "by_directory_bucket": dict(by_bucket.most_common(30)),
            "top_files": candidate_files,
        },
        "findings": [asdict(finding) for finding in findings],
    }


def _write_markdown(report: dict[str, object], output_path: Path) -> None:
    findings = report["findings"]
    summary = report["summary"]
    lines: list[str] = []
    lines.append("# Headless UI 风险扫描报告")
    lines.append("")
    lines.append(f"- 生成时间(UTC): `{report['generated_at_utc']}`")
    lines.append(f"- host 工程: `{report['host_project']}`")
    lines.append(f"- 实际编译文件数: `{report['effective_compile_file_count']}`")
    lines.append(f"- 风险条目数: `{report['finding_count']}`")
    lines.append(f"- 扫描范围: `{report['scope']}`")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    for label, count in summary["by_priority"].items():
        lines.append(f"- {label}: `{count}`")
    lines.append("")
    lines.append("## 建议处理")
    lines.append("")
    for label, count in summary["by_mitigation"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.append("")
    lines.append("## 高频目录")
    lines.append("")
    for bucket, count in summary["by_directory_bucket"].items():
        lines.append(f"- `{bucket}`: `{count}`")
    lines.append("")
    lines.append("## 高频文件")
    lines.append("")
    for entry in summary["top_files"][:20]:
        lines.append(f"- `{entry['count']}` 条 `{entry['mitigation']}`: `{entry['file']}`")
    lines.append("")
    lines.append("## P1 明细")
    lines.append("")
    p1_findings = [finding for finding in findings if finding["priority"] == 1]
    if not p1_findings:
        lines.append("- 无")
    else:
        for finding in p1_findings[:120]:
            member = f" member=`{finding['member_hint']}`" if finding["member_hint"] else ""
            lines.append(
                f"- `{finding['pattern_id']}` `{finding['file']}:{finding['line']}` `{finding['mitigation']}`{member}"
            )
            lines.append(f"  `{finding['code']}`")
    lines.append("")
    lines.append("## P2 明细")
    lines.append("")
    p2_findings = [finding for finding in findings if finding["priority"] == 2]
    if not p2_findings:
        lines.append("- 无")
    else:
        for finding in p2_findings[:120]:
            member = f" member=`{finding['member_hint']}`" if finding["member_hint"] else ""
            lines.append(
                f"- `{finding['pattern_id']}` `{finding['file']}:{finding['line']}` `{finding['mitigation']}`{member}"
            )
            lines.append(f"  `{finding['code']}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="扫描报告输出目录。",
    )
    parser.add_argument(
        "--print-top",
        type=int,
        default=20,
        help="终端里打印的高频文件数量。",
    )
    parser.add_argument(
        "--scope",
        choices=("compiled-models", "compiled-monsters", "compiled-all"),
        default="compiled-models",
        help="扫描范围。默认只看 host 实际编译到的 Core/Models 层。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not HOST_PROJECT.exists():
        raise FileNotFoundError(f"找不到 host 工程: {HOST_PROJECT}")
    compiled_files = [path for path in _expand_compile_nodes(HOST_PROJECT) if _in_scope(path, args.scope)]
    findings: list[Finding] = []
    for path in compiled_files:
        if path.suffix.lower() != ".cs":
            continue
        if not path.exists():
            continue
        findings.extend(_scan_file(path))
    findings.sort(key=lambda item: (item.priority, item.file, item.line, item.pattern_id))
    report = _build_report(compiled_files, findings)
    report["scope"] = args.scope
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"headless_ui_risk_scan_20260418_{args.scope.replace('-', '_')}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"[scan_headless_ui_risks] compiled_files={report['effective_compile_file_count']} findings={report['finding_count']}")
    top_files = report["summary"]["top_files"][: args.print_top]
    for entry in top_files:
        print(f"{entry['count']:>3}  {entry['file']}")
    print(f"[scan_headless_ui_risks] json={json_path}")
    print(f"[scan_headless_ui_risks] md={md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
