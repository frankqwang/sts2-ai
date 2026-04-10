#!/usr/bin/env python3
"""
HeadlessSim Stub Generator

Scans the STS2 source code to automatically generate stub files for types
that are excluded from the headless build but referenced by included code.

Similar approach to Sts2Repairer: source-scan based, no build required.

Usage:
    python stub_generator.py [project_root]
"""
from __future__ import annotations

import re
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import defaultdict

# ── Configuration ──

EXCLUDED_DIR_NAMES = {".git", ".godot", ".idea", ".vs", ".vscode", "bin", "obj"}

# Repo-tracked files excluded from the HeadlessSim build even though they live
# outside the excluded directory list above.
EXCLUDED_SOURCE_FILES = {
    "src/Core/Simulation/FullRunUiShim.cs",
}

# Directories excluded from HeadlessSim build
EXCLUDED_SOURCE_DIRS = [
    "src/Core/Nodes",
    "src/Core/Audio",
    "src/Core/Debug",
    "src/Core/Bindings",
    "src/Core/Animation",
    "src/Core/RichTextTags",
    "src/Core/DevConsole",
    "src/Core/AutoSlay",
    "src/Core/ControllerInput",
    "src/Core/Training",
    "src/Core/Multiplayer/Transport/Steam",
    "src/Core/Multiplayer/Transport/ENet",
    "src/Core/Platform",
    "STS2MCP",
    "MegaCrit",
]

# Files re-included (these are compiled, don't need stubs)
RE_INCLUDED_FILES = {
    "src/Core/Animation/AnimState.cs",
    "src/Core/Audio/DamageSfxType.cs",
    "src/Core/Audio/Debug/PitchVariance.cs",
    "src/Core/Audio/Debug/TmpSfx.cs",
    "src/Core/Audio/FmodSfx.cs",
    "src/Core/Nodes/Combat/TargetMode.cs",
    "src/Core/Nodes/CommonUi/CardPreviewStyle.cs",
    "src/Core/Nodes/Events/ICustomEventNode.cs",
    "src/Core/Nodes/Pooling/INodePool.cs",
    "src/Core/Nodes/Pooling/IPoolable.cs",
    "src/Core/Nodes/Rooms/IRoomWithProceedButton.cs",
    "src/Core/Nodes/Screens/Capstones/ICapstoneScreen.cs",
    "src/Core/Nodes/Screens/CapstoneSubmenuType.cs",
    "src/Core/Nodes/Screens/CardSelection/ICardSelector.cs",
    "src/Core/Nodes/Screens/CharacterSelect/ICharacterSelectButtonDelegate.cs",
    "src/Core/Nodes/Screens/FeedbackScreen/FeedbackData.cs",
    "src/Core/Nodes/Screens/Map/DrawingMode.cs",
    "src/Core/Nodes/Screens/Overlays/IOverlayScreen.cs",
    "src/Core/Nodes/Screens/RunHistoryScreen/GameOverType.cs",
    "src/Core/Nodes/Screens/Timeline/EpochComparer.cs",
    "src/Core/Nodes/Screens/Timeline/EpochSlotState.cs",
    "src/Core/Nodes/Vfx/IDeathDelayer.cs",
    "src/Core/Nodes/Vfx/Utilities/DialogueSide.cs",
    "src/Core/Nodes/Vfx/Utilities/DialogueStyle.cs",
    "src/Core/Nodes/Vfx/Utilities/RumbleStyle.cs",
    "src/Core/Nodes/Vfx/Utilities/ShakeDuration.cs",
    "src/Core/Nodes/Vfx/Utilities/ShakeStrength.cs",
    "src/Core/Nodes/Vfx/VfxColor.cs",
    "src/Core/Nodes/Vfx/VfxPosition.cs",
    "src/Core/Debug/CustomDateTimeConverter.cs",
    "src/Core/Debug/DebugSettings.cs",
    "src/Core/Debug/ReleaseInfo.cs",
    "src/Core/Training/CombatTrainingDtos.cs",
    "src/Core/Training/CombatTrainingActionType.cs",
    "src/Core/Training/CombatTrainingActionRequest.cs",
    "src/Core/Training/CombatTrainingResetRequest.cs",
    "src/Core/Training/CombatTrainingMode.cs",
    "src/Core/Training/CombatTrainingChoiceAdapter.cs",
    "src/Core/Training/CombatTrainingSimulatorChoiceBridge.cs",
    "src/Core/Training/CombatTrainingEnvService.cs",
    "src/Core/Platform/Null/NullLeaderboard.cs",
    "src/Core/Platform/Null/NullLeaderboardFile.cs",
    "src/Core/Platform/Null/NullLeaderboardFileEntry.cs",
    "src/Core/Platform/Null/NullMultiplayerName.cs",
    "src/Core/Platform/Steam/SteamDisconnectionReason.cs",
    "src/Core/Platform/Steam/SteamDisconnectionReasonExtensions.cs",
}

# Types whose inheritance chain should be stripped to avoid CS0533/CS0534 cascading.
# The class is still generated (so references compile) but without inheriting abstract bases.
STRIP_INHERITANCE_TYPES = {
    "NDeckCardSelectScreen",
    "NDeckEnchantSelectScreen",
    "NDeckTransformSelectScreen",
    "NDeckUpgradeSelectScreen",
    "NSimpleCardSelectScreen",
}

# ── Data Structures ──

@dataclass
class TypeMember:
    """A public member of a type (property, method, field, event, enum value)."""
    kind: str          # 'property', 'method', 'field', 'event', 'const', 'enum_value', 'constructor', 'indexer'
    name: str
    return_type: str   # for methods/properties
    is_static: bool
    is_abstract: bool
    is_virtual: bool
    is_override: bool
    params: str = ""   # for methods: "int x, string y"
    generic_params: str = ""  # for methods: "<T>" / "<TKey, TValue>"
    default_value: str = ""  # for fields/consts


@dataclass
class TypeInfo:
    """A parsed type definition."""
    name: str
    namespace: str
    kind: str          # 'class', 'enum', 'struct', 'interface'
    is_static: bool
    is_abstract: bool
    is_partial: bool
    base_types: list[str] = field(default_factory=list)  # base class + interfaces
    members: list[TypeMember] = field(default_factory=list)
    enum_values: list[str] = field(default_factory=list)  # for enums
    nested_types: list[TypeInfo] = field(default_factory=list)
    source_file: str = ""
    generic_params: str = ""  # e.g. "<T>"


@dataclass
class StubGeneratorResult:
    """Result of stub generation."""
    types_scanned: int = 0
    types_referenced: int = 0
    stubs_generated: int = 0
    warnings: list[str] = field(default_factory=list)


# ── Source Code Scanner ──

class CSharpScanner:
    """Scans C# source files to extract type definitions and their members."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        # Map of type_name -> namespace for ALL types in the project (included + excluded)
        self.all_type_namespaces: dict[str, set[str]] = defaultdict(set)
        # Map of type_name -> list of abstract members for abstract classes
        self.abstract_class_members: dict[str, list[TypeMember]] = {}
        # Namespaces that have types in INCLUDED code (safe to add as using directives)
        self.included_namespaces: set[str] = set()
        # Map of interface_name -> list of abstract members for interfaces
        self.interface_members: dict[str, list[TypeMember]] = {}

    def scan_excluded_types(self) -> dict[str, TypeInfo]:
        """Scan excluded directories and return all public type definitions."""
        types: dict[str, TypeInfo] = {}  # fully qualified name -> TypeInfo

        for excluded_dir in EXCLUDED_SOURCE_DIRS:
            dir_path = self.project_root / excluded_dir
            if not dir_path.exists():
                continue
            for cs_file in dir_path.rglob("*.cs"):
                if cs_file.suffix == ".uid":
                    continue
                # Skip re-included files
                rel = cs_file.relative_to(self.project_root).as_posix()
                if rel in RE_INCLUDED_FILES:
                    continue
                # Skip bin/obj
                parts = {p.lower() for p in cs_file.relative_to(self.project_root).parts}
                if parts & EXCLUDED_DIR_NAMES:
                    continue

                try:
                    content = cs_file.read_text(encoding="utf-8-sig")
                except Exception:
                    continue

                file_types = self._parse_types(content, rel)
                for t in file_types:
                    fqn = f"{t.namespace}.{t.name}"
                    # Keep the more complete definition if duplicate
                    if fqn not in types or len(t.members) > len(types[fqn].members):
                        types[fqn] = t

        return types

    def scan_all_type_namespaces(self):
        """Scan ALL source files (included + excluded) to build a type_name -> namespace map.
        Also discovers abstract class members for CS0534 fix and interface members for CS0535."""
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return

        for cs_file in src_dir.rglob("*.cs"):
            if cs_file.suffix == ".uid":
                continue
            parts_set = {p.lower() for p in cs_file.relative_to(self.project_root).parts}
            if parts_set & EXCLUDED_DIR_NAMES:
                continue
            try:
                content = cs_file.read_text(encoding="utf-8-sig")
            except Exception:
                continue

            ns_match = re.search(r'namespace\s+([\w.]+)', content)
            if not ns_match:
                continue
            namespace = ns_match.group(1)

            # Check if this file is in included code (not in excluded dirs, or re-included)
            rel = cs_file.relative_to(self.project_root).as_posix()
            is_included = True
            if rel in EXCLUDED_SOURCE_FILES:
                is_included = False
            for exc in EXCLUDED_SOURCE_DIRS:
                if rel.startswith(exc + "/"):
                    if rel not in RE_INCLUDED_FILES:
                        is_included = False
                    break

            if is_included:
                self.included_namespaces.add(namespace)

            # Find all type definitions and record namespace
            for m in re.finditer(
                r'(?:public|internal)\s+(?:(?:static|partial|abstract|sealed|readonly|new)\s+)*'
                r'(class|enum|struct|interface)\s+(\w+)', content
            ):
                kind = m.group(1)
                name = m.group(2)
                self.all_type_namespaces[name].add(namespace)

            # Find abstract classes and extract their abstract members
            abs_pattern = re.compile(
                r'(?:public|internal)\s+abstract\s+(?:partial\s+)?(?:class)\s+(\w+)'
                r'(?:\s*<[^>]+>)?'
                r'[^{]*\{',
                re.MULTILINE
            )
            for am in abs_pattern.finditer(content):
                class_name = am.group(1)
                brace_start = am.end() - 1
                body_end = self._find_matching_brace(content, brace_start)
                if body_end == -1:
                    continue
                body = content[brace_start + 1:body_end]
                abstract_members = self._parse_abstract_members(body)
                if abstract_members:
                    self.abstract_class_members[class_name] = abstract_members

            # Find interfaces and extract their members (for CS0535 fix)
            iface_pattern = re.compile(
                r'(?:public|internal)\s+(?:partial\s+)?interface\s+(\w+)'
                r'(?:\s*<[^>]+>)?'
                r'[^{]*\{',
                re.MULTILINE
            )
            for im in iface_pattern.finditer(content):
                iface_name = im.group(1)
                brace_start = im.end() - 1
                body_end = self._find_matching_brace(content, brace_start)
                if body_end == -1:
                    continue
                body = content[brace_start + 1:body_end]
                iface_members = self._parse_interface_members(body)
                if iface_members:
                    if iface_name in self.interface_members:
                        # Merge (partial interfaces)
                        existing_names = {m.name for m in self.interface_members[iface_name]}
                        for nm in iface_members:
                            if nm.name not in existing_names:
                                self.interface_members[iface_name].append(nm)
                    else:
                        self.interface_members[iface_name] = iface_members

        # Also scan MegaCrit directory
        megacrit_dir = self.project_root / "MegaCrit"
        if megacrit_dir.exists():
            for cs_file in megacrit_dir.rglob("*.cs"):
                if cs_file.suffix == ".uid":
                    continue
                parts_set = {p.lower() for p in cs_file.relative_to(self.project_root).parts}
                if parts_set & EXCLUDED_DIR_NAMES:
                    continue
                try:
                    content = cs_file.read_text(encoding="utf-8-sig")
                except Exception:
                    continue
                ns_match = re.search(r'namespace\s+([\w.]+)', content)
                if not ns_match:
                    continue
                namespace = ns_match.group(1)
                for m in re.finditer(
                    r'(?:public|internal)\s+(?:(?:static|partial|abstract|sealed|readonly|new)\s+)*'
                    r'(class|enum|struct|interface)\s+(\w+)', content
                ):
                    self.all_type_namespaces[m.group(2)].add(namespace)

        # Also scan GodotSharpStub and HeadlessSim stub files
        # This ensures Godot types (Control, Node, Vector2...) and third-party types
        # (MegaLabel, Steamworks types...) are in the type registry
        extra_dirs = [
            self.project_root / "tools" / "headless-sim" / "GodotSharpStub" / "src",
            self.project_root / "tools" / "headless-sim" / "HeadlessSim",
        ]
        for extra_dir in extra_dirs:
            if not extra_dir.exists():
                continue
            for cs_file in extra_dir.rglob("*.cs"):
                try:
                    content = cs_file.read_text(encoding="utf-8-sig")
                except Exception:
                    continue
                ns_match = re.search(r'namespace\s+([\w.]+)', content)
                if not ns_match:
                    continue
                namespace = ns_match.group(1)
                self.included_namespaces.add(namespace)
                for m in re.finditer(
                    r'(?:public|internal)\s+(?:(?:static|partial|abstract|sealed|readonly|new)\s+)*'
                    r'(class|enum|struct|interface)\s+(\w+)', content
                ):
                    self.all_type_namespaces[m.group(2)].add(namespace)

    def _parse_abstract_members(self, body: str) -> list[TypeMember]:
        """Parse abstract members from a class body."""
        members = []
        # Remove nested types
        clean = self._remove_nested_types(body)
        # Remove comments
        clean = re.sub(r'//.*$', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'/\*.*?\*/', '', clean, flags=re.DOTALL)

        # Abstract properties (track access modifier via default_value field)
        for m in re.finditer(
            r'(public|protected)\s+abstract\s+([\w<>\[\]?,.\s]+?)\s+(\w+)\s*\{',
            clean
        ):
            access = m.group(1)
            ret_type = m.group(2).strip()
            name = m.group(3)
            members.append(TypeMember(
                kind='property', name=name, return_type=ret_type,
                is_static=False, is_abstract=True, is_virtual=False, is_override=False,
                default_value=access,  # store access modifier
            ))

        # Abstract methods (track access modifier via default_value field)
        for m in re.finditer(
            r'(public|protected)\s+abstract\s+([\w<>\[\]?,.\s]+?)\s+(\w+)(\s*<[^(){};=>]+>)?\s*\(([^)]*)\)\s*;',
            clean
        ):
            access = m.group(1)
            ret_type = m.group(2).strip()
            name = m.group(3)
            generic_params = (m.group(4) or "").strip()
            params = m.group(5).strip()
            members.append(TypeMember(
                kind='method', name=name, return_type=ret_type, params=params,
                generic_params=generic_params,
                is_static=False, is_abstract=True, is_virtual=False, is_override=False,
                default_value=access,  # store access modifier
            ))

        return members

    def _parse_interface_members(self, body: str) -> list[TypeMember]:
        """Parse members from an interface body."""
        members = []
        # Remove nested types
        clean = self._remove_nested_types(body)
        # Remove comments
        clean = re.sub(r'//.*$', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'/\*.*?\*/', '', clean, flags=re.DOTALL)

        # Interface properties: Type Name { get; set; }
        for m in re.finditer(
            r'(?:public\s+)?([\w<>\[\]?,.\s]+?)\s+(\w+)\s*\{[^}]*\}',
            clean
        ):
            ret_type = m.group(1).strip()
            name = m.group(2)
            if name in ('get', 'set', 'init', 'add', 'remove'):
                continue
            # Skip if it looks like a method body
            if ret_type in ('if', 'else', 'for', 'while', 'switch', 'return', 'var'):
                continue
            members.append(TypeMember(
                kind='property', name=name, return_type=ret_type,
                is_static=False, is_abstract=True, is_virtual=False, is_override=False,
            ))

        # Interface methods: RetType Name(params);
        for m in re.finditer(
            r'(?:public\s+)?([\w<>\[\]?,.\s]+?)\s+(\w+)(\s*<[^(){};=>]+>)?\s*\(([^)]*)\)\s*;',
            clean
        ):
            ret_type = m.group(1).strip()
            name = m.group(2)
            generic_params = (m.group(3) or "").strip()
            params = m.group(4).strip()
            members.append(TypeMember(
                kind='method', name=name, return_type=ret_type, params=params,
                generic_params=generic_params,
                is_static=False, is_abstract=True, is_virtual=False, is_override=False,
            ))

        return members

    def scan_references_in_included_code(self, excluded_type_names: set[str]) -> set[str]:
        """Scan included code to find which excluded type names are referenced."""
        referenced = set()

        for cs_file in self._iter_included_files():
            try:
                content = cs_file.read_text(encoding="utf-8-sig")
            except Exception:
                continue

            for type_name in excluded_type_names:
                # Simple word-boundary match
                if re.search(rf'\b{re.escape(type_name)}\b', content):
                    referenced.add(type_name)

        return referenced

    def _iter_included_files(self):
        """Iterate over files that ARE included in the HeadlessSim build."""
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return

        for cs_file in src_dir.rglob("*.cs"):
            if cs_file.suffix == ".uid":
                continue
            parts = {p.lower() for p in cs_file.relative_to(self.project_root).parts}
            if parts & EXCLUDED_DIR_NAMES:
                continue

            # Check if file is in an excluded directory
            rel = cs_file.relative_to(self.project_root).as_posix()
            if rel in EXCLUDED_SOURCE_FILES:
                continue
            in_excluded = False
            for exc in EXCLUDED_SOURCE_DIRS:
                if rel.startswith(exc + "/"):
                    # Check if re-included
                    if rel not in RE_INCLUDED_FILES:
                        in_excluded = True
                    break
            if not in_excluded:
                yield cs_file

    def _parse_types(self, content: str, source_file: str) -> list[TypeInfo]:
        """Parse all public type definitions from a C# source file."""
        types = []

        # Extract namespace
        ns_match = re.search(r'namespace\s+([\w.]+)', content)
        if not ns_match:
            return types
        namespace = ns_match.group(1)

        # Find type definitions
        type_pattern = re.compile(
            r'(?:^|\n)\s*'
            r'(public\s+(?:(?:static|partial|abstract|sealed|readonly|new)\s+)*)'
            r'(class|enum|struct|interface)\s+'
            r'(\w+)'
            r'(\s*<[^>]+>)?'  # generic params
            r'([^{]*?)'       # base types
            r'\{',
            re.MULTILINE
        )

        for m in type_pattern.finditer(content):
            modifiers = m.group(1)
            kind = m.group(2)
            name = m.group(3)
            generic_params = (m.group(4) or "").strip()
            inheritance = m.group(5).strip()

            # Parse modifiers
            is_static = 'static' in modifiers
            is_abstract = 'abstract' in modifiers
            is_partial = 'partial' in modifiers

            # Parse base types
            base_types = []
            if inheritance.startswith(':'):
                bases = inheritance[1:].strip()
                # Split by comma, handling generic types
                depth = 0
                current = ""
                for ch in bases:
                    if ch in '<':
                        depth += 1
                    elif ch in '>':
                        depth -= 1
                    elif ch == ',' and depth == 0:
                        bt = current.strip()
                        if bt:
                            base_types.append(bt)
                        current = ""
                        continue
                    current += ch
                bt = current.strip()
                if bt:
                    base_types.append(bt)

            # Find the body of the type
            brace_start = m.end() - 1
            body_end = self._find_matching_brace(content, brace_start)
            if body_end == -1:
                continue

            body = content[brace_start + 1:body_end]

            type_info = TypeInfo(
                name=name,
                namespace=namespace,
                kind=kind,
                is_static=is_static,
                is_abstract=is_abstract,
                is_partial=is_partial,
                base_types=base_types,
                source_file=source_file,
                generic_params=generic_params,
            )

            if kind == 'enum':
                type_info.enum_values = self._parse_enum_values(body)
            else:
                type_info.members = self._parse_members(body, name)

            types.append(type_info)

        return types

    def _find_matching_brace(self, text: str, opening_brace_index: int) -> int:
        """Find the matching closing brace."""
        depth = 0
        in_string = False
        in_char = False
        in_line_comment = False
        in_block_comment = False
        i = opening_brace_index

        while i < len(text):
            ch = text[i]
            prev = text[i - 1] if i > 0 else ''
            next_ch = text[i + 1] if i + 1 < len(text) else ''

            if in_line_comment:
                if ch == '\n':
                    in_line_comment = False
            elif in_block_comment:
                if ch == '*' and next_ch == '/':
                    in_block_comment = False
                    i += 1
            elif in_string:
                if ch == '"' and prev != '\\':
                    in_string = False
            elif in_char:
                if ch == "'" and prev != '\\':
                    in_char = False
            else:
                if ch == '/' and next_ch == '/':
                    in_line_comment = True
                elif ch == '/' and next_ch == '*':
                    in_block_comment = True
                elif ch == '"':
                    in_string = True
                elif ch == "'":
                    in_char = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return i

            i += 1
        return -1

    def _parse_enum_values(self, body: str) -> list[str]:
        """Parse enum values from an enum body."""
        values = []
        # Remove comments
        body = re.sub(r'//.*$', '', body, flags=re.MULTILINE)
        body = re.sub(r'/\*.*?\*/', '', body, flags=re.DOTALL)

        for line in body.split(','):
            line = line.strip()
            if not line:
                continue
            # Match "Name" or "Name = Value"
            m = re.match(r'(\w+)(?:\s*=\s*(.+))?', line)
            if m:
                name = m.group(1)
                value = m.group(2)
                if value:
                    values.append(f"{name} = {value.strip()}")
                else:
                    values.append(name)
        return values

    def _parse_members(self, body: str, class_name: str) -> list[TypeMember]:
        """Parse public members from a class/struct/interface body."""
        members = []

        # Remove nested type bodies to avoid parsing their members
        clean_body = self._remove_nested_types(body)

        # Remove comments
        clean_body = re.sub(r'//.*$', '', clean_body, flags=re.MULTILINE)
        clean_body = re.sub(r'/\*.*?\*/', '', clean_body, flags=re.DOTALL)

        # Find properties (public and protected - needed for abstract override chain)
        # Match both { get; set; } style and => expression body style
        prop_pattern = re.compile(
            r'(?:public|protected)\s+'
            r'((?:(?:static|override|virtual|abstract|new|readonly|required)\s+)*)'
            r'([\w<>\[\]?,.\s]+?)\s+'
            r'(\w+)\s*(?:\{|=>)'
        )
        for m in prop_pattern.finditer(clean_body):
            modifiers = m.group(1)
            ret_type = m.group(2).strip()
            name = m.group(3)
            if name in ('get', 'set', 'init', 'value', 'add', 'remove'):
                continue
            members.append(TypeMember(
                kind='property',
                name=name,
                return_type=ret_type,
                is_static='static' in modifiers,
                is_abstract='abstract' in modifiers,
                is_virtual='virtual' in modifiers,
                is_override='override' in modifiers,
            ))

        # Find methods - use non-greedy and limit to reasonable param length
        method_pattern = re.compile(
            r'public\s+'
            r'((?:(?:static|override|virtual|abstract|new|async)\s+)*)'
            r'([\w<>\[\]?,.\s]+?)\s+'
            r'(\w+)(\s*<[^(){};=>]+>)?\s*\(([^)]{0,500})\)'
        )
        for m in method_pattern.finditer(clean_body):
            modifiers = m.group(1)
            ret_type = m.group(2).strip()
            name = m.group(3)
            generic_params = (m.group(4) or "").strip()
            params = m.group(5).strip()
            # Skip property accessors that look like methods
            if name in ('get', 'set', 'add', 'remove'):
                continue
            # Skip delegate declarations (they're type declarations, not methods)
            if 'delegate' in ret_type:
                continue
            members.append(TypeMember(
                kind='method',
                name=name,
                return_type=ret_type,
                is_static='static' in modifiers,
                is_abstract='abstract' in modifiers,
                is_virtual='virtual' in modifiers,
                is_override='override' in modifiers,
                params=params,
                generic_params=generic_params,
            ))

        # Find constructors
        ctor_pattern = re.compile(
            rf'public\s+{re.escape(class_name)}\s*\(([^)]*)\)'
        )
        for m in ctor_pattern.finditer(clean_body):
            params = m.group(1).strip()
            if params:
                members.append(TypeMember(
                    kind='constructor',
                    name=class_name,
                    return_type='',
                    is_static=False,
                    is_abstract=False,
                    is_virtual=False,
                    is_override=False,
                    params=params,
                ))

        # Find events
        event_pattern = re.compile(
            r'public\s+'
            r'((?:(?:static|new)\s+)*)'
            r'event\s+([\w<>?.]+)\s+(\w+)\s*;'
        )
        for m in event_pattern.finditer(clean_body):
            modifiers = m.group(1)
            event_type = m.group(2)
            name = m.group(3)
            members.append(TypeMember(
                kind='event',
                name=name,
                return_type=event_type,
                is_static='static' in modifiers,
                is_abstract=False,
                is_virtual=False,
                is_override=False,
            ))

        # Find const/static readonly fields
        field_pattern = re.compile(
            r'public\s+'
            r'((?:(?:static|readonly|const|new)\s+)*)'
            r'([\w<>\[\]?,.\s]+?)\s+'
            r'(\w+)\s*[;=]'
        )
        for m in field_pattern.finditer(clean_body):
            modifiers = m.group(1)
            field_type = m.group(2).strip()
            name = m.group(3)
            if 'const' in modifiers or ('static' in modifiers and 'readonly' in modifiers):
                members.append(TypeMember(
                    kind='const' if 'const' in modifiers else 'field',
                    name=name,
                    return_type=field_type,
                    is_static=True,
                    is_abstract=False,
                    is_virtual=False,
                    is_override=False,
                ))

        return members

    def _remove_nested_types(self, body: str) -> str:
        """Remove nested class/struct/enum definitions to avoid parsing their members."""
        result = body
        # Iteratively remove nested type blocks
        pattern = re.compile(
            r'(?:public|private|protected|internal)\s+(?:(?:static|partial|abstract|sealed|new|readonly)\s+)*'
            r'(?:class|struct|enum|interface)\s+\w+[^{]*\{',
        )
        for _ in range(10):  # max depth
            m = pattern.search(result)
            if not m:
                break
            brace_start = result.rfind('{', m.start(), m.end())
            if brace_start == -1:
                break
            brace_end = self._find_matching_brace(result, brace_start)
            if brace_end == -1:
                break
            result = result[:m.start()] + result[brace_end + 1:]
        return result


# ── Stub Code Generator ──

class StubCodeGenerator:
    """Generates C# stub code from TypeInfo objects."""

    # Types that map to Godot base classes
    GODOT_BASE_MAP = {
        'Node', 'Node2D', 'Node3D', 'Control', 'Resource', 'GodotObject',
        'RefCounted', 'Sprite2D', 'TextureRect', 'Label', 'RichTextLabel',
        'Panel', 'Button', 'LineEdit', 'TextEdit', 'ColorRect',
        'AnimatedSprite2D', 'Area2D', 'Camera2D', 'CanvasLayer',
        'GpuParticles2D', 'BackBufferCopy', 'SubViewport',
    }

    def generate_stub_file(self, types_by_ns: dict[str, list[TypeInfo]],
                           referenced_names: set[str],
                           all_type_namespaces: dict[str, set[str]] | None = None,
                           abstract_class_members: dict[str, list[TypeMember]] | None = None,
                           interface_members: dict[str, list[TypeMember]] | None = None,
                           included_namespaces: set[str] | None = None) -> str:
        """Generate a complete stub C# file."""
        self._all_type_namespaces = all_type_namespaces or {}
        self._abstract_class_members = abstract_class_members or {}
        self._interface_members = interface_members or {}

        # Build a name->TypeInfo lookup for inheritance chain traversal
        self._types_by_name: dict[str, TypeInfo] = {}
        for ns_types in types_by_ns.values():
            for t in ns_types:
                self._types_by_name[t.name] = t

        # Build set of all known type names (from stubs + included source)
        self._all_stub_type_names = set()
        for ns_types in types_by_ns.values():
            for t in ns_types:
                if t.name in referenced_names:
                    self._all_stub_type_names.add(t.name)
        # Add types known from scanning all source
        if all_type_namespaces:
            self._all_stub_type_names.update(all_type_namespaces.keys())

        # Collect namespaces safe to add as using directives
        # Only include namespaces that exist in included code or the generated stubs
        safe_namespaces = included_namespaces or set()
        all_usings = {
            "System",
            "System.Collections.Generic",
            "System.Linq",
            "System.Threading",
            "System.Threading.Tasks",
            "Godot",
        }
        # Add namespaces from the stubs themselves
        for ns in types_by_ns.keys():
            all_usings.add(ns)
        # Add namespaces from included source (these are safe -- they exist in the build)
        all_usings.update(safe_namespaces)

        lines = [
            "// AUTO-GENERATED by stub_generator.py",
            "// Stubs for excluded types referenced by HeadlessSim code.",
            "// Re-run the generator when game source is updated.",
            "//",
            "// DO NOT EDIT MANUALLY.",
            "",
            "#nullable enable",
            "#pragma warning disable CS0414, CS0649, CS0108, CS0114, CS0109, CS8618, CS0067, CS0105, CS0115",
            "",
        ]
        for ns in sorted(all_usings):
            lines.append(f"using {ns};")
        lines.append("")

        for ns in sorted(types_by_ns.keys()):
            ns_types = types_by_ns[ns]
            # Filter to only referenced types
            relevant = [t for t in ns_types if t.name in referenced_names]
            if not relevant:
                continue

            lines.append(f"namespace {ns}")
            lines.append("{")

            for type_info in sorted(relevant, key=lambda t: t.name):
                stub_lines = self._generate_type_stub(type_info)
                for sl in stub_lines:
                    lines.append(f"    {sl}")

            lines.append("}")
            lines.append("")

        return "\n".join(lines)

    def _generate_type_stub(self, t: TypeInfo) -> list[str]:
        """Generate stub code for a single type."""
        if t.kind == 'enum':
            return self._generate_enum_stub(t)
        elif t.kind == 'interface':
            return self._generate_interface_stub(t)
        else:
            return self._generate_class_stub(t)

    def _generate_enum_stub(self, t: TypeInfo) -> list[str]:
        """Generate an enum stub."""
        values = t.enum_values if t.enum_values else ["None"]
        vals_str = ", ".join(values)
        return [f"public enum {t.name} {{ {vals_str} }}"]

    def _generate_interface_stub(self, t: TypeInfo) -> list[str]:
        """Generate an interface stub."""
        bases = ""
        if t.base_types:
            bases = " : " + ", ".join(t.base_types)
        return [f"public partial interface {t.name}{t.generic_params}{bases} {{ }}"]

    def _generate_class_stub(self, t: TypeInfo) -> list[str]:
        """Generate a class/struct stub with members."""
        lines = []
        strip_inheritance = t.name in STRIP_INHERITANCE_TYPES

        # Build declaration
        modifiers = []
        if t.is_static:
            modifiers.append("static")
        if t.is_abstract:
            modifiers.append("abstract")
        modifiers.append("partial")

        mod_str = " ".join(modifiers) + " " if modifiers else ""

        # Base types
        base_str = ""
        base_types = list(t.base_types)
        if strip_inheritance:
            keep = []
            for bt in base_types:
                simple = bt.split('<')[0].split('.')[-1].strip()
                if simple in self.GODOT_BASE_MAP:
                    keep.append(f"Godot.{bt}")
            keep.extend(self._collect_interface_bases(t))
            if not keep:
                keep.append("Godot.Control")
            base_str = " : " + ", ".join(dict.fromkeys(keep))
        elif base_types:
            mapped_bases = []
            for bt in base_types:
                # Map Godot types to full name
                simple = bt.split('<')[0].strip()
                if simple in self.GODOT_BASE_MAP:
                    mapped_bases.append(f"Godot.{bt}")
                else:
                    mapped_bases.append(bt)
            base_str = " : " + ", ".join(mapped_bases)

        lines.append(f"public {mod_str}{t.kind} {t.name}{t.generic_params}{base_str}")
        lines.append("{")

        # Generate members
        existing_names = set()
        for member in t.members:
            member_line = self._generate_member_stub(member, t)
            if member_line:
                lines.append(f"    {member_line}")
                existing_names.add(member.name)

        # For abstract stub classes: re-declare abstract members from base classes
        # so that child stubs can override them (CS0534 / CS0115 fix)
        if t.is_abstract and t.base_types and not strip_inheritance:
            inherited_abstract = self._collect_abstract_members(t, set(existing_names))
            for am in inherited_abstract:
                line = self._generate_abstract_redeclaration(am)
                if line:
                    lines.append(f"    {line}")
                    existing_names.add(am.name)

        # For stripped UI stubs: keep interface contracts without inheriting the heavy base class.
        if strip_inheritance:
            interface_members = self._collect_interface_members(t, existing_names)
            for im in interface_members:
                line = self._generate_abstract_override(im)
                if line:
                    lines.append(f"    {line}")
                    existing_names.add(im.name)

        # For non-abstract stub classes: generate override implementations (CS0534 fix)
        if not t.is_abstract and t.base_types and not strip_inheritance:
            abstract_members = self._collect_abstract_members(t, existing_names)
            for am in abstract_members:
                line = self._generate_abstract_override(am)
                if line:
                    lines.append(f"    {line}")

        lines.append("}")
        return lines

    def _collect_abstract_members(self, t: TypeInfo, existing_names: set[str]) -> list[TypeMember]:
        """Collect abstract members from base classes and interfaces that need implementations.
        Traverses the full inheritance chain (including stubs extending other stubs)."""
        result = []
        abstract_members = getattr(self, '_abstract_class_members', {})
        interface_members = getattr(self, '_interface_members', {})

        # BFS through the inheritance chain
        visited_bases = set()
        bases_to_check = list(t.base_types)

        while bases_to_check:
            bt = bases_to_check.pop(0)
            base_simple = bt.split('<')[0].split('.')[-1].strip()
            if base_simple in visited_bases:
                continue
            visited_bases.add(base_simple)

            # Check abstract class members
            if base_simple in abstract_members:
                for am in abstract_members[base_simple]:
                    if am.name not in existing_names:
                        result.append(am)
                        existing_names.add(am.name)

            # Check interface members
            if base_simple in interface_members:
                for im in interface_members[base_simple]:
                    if im.name not in existing_names:
                        result.append(TypeMember(
                            kind=im.kind, name=im.name, return_type=im.return_type,
                            params=im.params, is_static=False, is_abstract=False,
                            is_virtual=False, is_override=False,
                            default_value='interface',
                        ))
                        existing_names.add(im.name)

            # Look up this base type's own bases for further traversal
            all_type_ns = getattr(self, '_all_type_namespaces', {})
            # Find this base in excluded_types to get its base_types
            for fqn_key, type_info in self._types_by_name.items():
                if type_info.name == base_simple:
                    for parent_bt in type_info.base_types:
                        bases_to_check.append(parent_bt)
                    break

        return result

    def _generate_abstract_redeclaration(self, m: TypeMember) -> Optional[str]:
        """Re-declare an abstract member in an abstract stub class so child classes
        can override it."""
        generic_type_names = self._extract_generic_param_names(m.generic_params)
        ret = self._simplify_return_type(m.return_type, generic_type_names)
        access = m.default_value if m.default_value in ('public', 'protected') else 'public'
        if m.kind == 'property':
            return f"{access} abstract {ret} {m.name} {{ get; }}"
        elif m.kind == 'method':
            params = self._simplify_params(m.params, extra_known_types=generic_type_names)
            return f"{access} abstract {ret} {m.name}{m.generic_params}({params});"
        return None

    def _generate_abstract_override(self, m: TypeMember) -> Optional[str]:
        """Generate an override implementation for an abstract/interface member."""
        generic_type_names = self._extract_generic_param_names(m.generic_params)
        ret = self._simplify_return_type(m.return_type, generic_type_names)
        access = m.default_value if m.default_value in ('public', 'protected') else 'public'
        is_interface = m.default_value == 'interface'

        if is_interface:
            # Interface implementation: public, no override keyword
            if m.kind == 'property':
                return f"public {ret} {m.name} => default!;"
            elif m.kind == 'method':
                params = self._simplify_params(m.params, extra_known_types=generic_type_names)
                body = self._generate_method_body(ret)
                return f"public {ret} {m.name}{m.generic_params}({params}){body}"
        else:
            # Abstract class override: use override + match access modifier
            if m.kind == 'property':
                return f"{access} override {ret} {m.name} => default!;"
            elif m.kind == 'method':
                params = self._simplify_params(m.params, extra_known_types=generic_type_names)
                body = self._generate_method_body(ret)
                return f"{access} override {ret} {m.name}{m.generic_params}({params}){body}"
        return None

    def _generate_member_stub(self, m: TypeMember, parent: TypeInfo) -> Optional[str]:
        """Generate a stub for a single member."""
        static = "static " if m.is_static else ""
        strip_inheritance = parent.name in STRIP_INHERITANCE_TYPES
        # Keep 'override' for members that implement known abstract members.
        # Use 'new' only for non-abstract overrides that may not exist in stub base class.
        override = ""
        if m.is_override and not strip_inheritance:
            # Check if this member implements a known abstract member from base class
            known_abstract = False
            abstract_members = getattr(self, '_abstract_class_members', {})
            for bt in parent.base_types:
                base_simple = bt.split('<')[0].split('.')[-1].strip()
                if base_simple in abstract_members:
                    for am in abstract_members[base_simple]:
                        if am.name == m.name:
                            known_abstract = True
                            break
            if known_abstract:
                override = "override "
            else:
                override = "new "
        virtual = "virtual " if m.is_virtual and not m.is_override else ""
        abstract = "abstract " if m.is_abstract else ""

        if m.kind == 'property':
            ret = self._simplify_return_type(
                "IEnumerable<string>" if m.name in ("AssetPaths", "AllAssetPaths", "assetPaths") else m.return_type
            )
            if m.is_abstract:
                return f"public {abstract}{static}{ret} {m.name} {{ get; }}"
            return f"public {override}{virtual}{static}{ret} {m.name} {{ get; set; }}"

        elif m.kind == 'method':
            generic_type_names = self._extract_generic_param_names(m.generic_params)
            ret = self._simplify_return_type(m.return_type, generic_type_names)
            params = self._simplify_params(m.params, extra_known_types=generic_type_names)
            body = self._generate_method_body(ret, params)
            if m.is_abstract:
                return f"public {abstract}{static}{ret} {m.name}{m.generic_params}({params});"
            return f"public {override}{virtual}{static}{ret} {m.name}{m.generic_params}({params}){body}"

        elif m.kind == 'constructor':
            params = self._simplify_params(m.params)
            ctor_base = ""
            parent_bases = [bt.split('<')[0].split('.')[-1].strip() for bt in parent.base_types]
            if "NetClient" in parent_bases and "INetClientHandler handler" in params:
                ctor_base = " : base(handler)"
            elif "NetHost" in parent_bases and "INetHostHandler handler" in params:
                ctor_base = " : base(handler)"
            return f"public {parent.name}({params}){ctor_base} {{ }}"

        elif m.kind == 'event':
            evt_type = m.return_type.rstrip('?') + '?'
            return f"public {static}event {evt_type} {m.name};"

        elif m.kind == 'const':
            default = self._default_for_type(m.return_type)
            return f"public const {m.return_type} {m.name} = {default};"

        elif m.kind == 'field':
            default = self._default_for_type(m.return_type)
            return f"public {static}readonly {m.return_type} {m.name} = {default};"

        return None

    # Types that are known to be available in the headless build
    # This is a fallback - most types are discovered via scan_all_type_namespaces()
    KNOWN_TYPES = {
        # C# primitives
        'void', 'bool', 'int', 'float', 'double', 'long', 'ulong', 'short', 'byte', 'uint',
        'string', 'object', 'dynamic', 'decimal', 'char', 'nint', 'nuint',
        # System collections and tasks
        'Task', 'IEnumerable', 'IReadOnlyList', 'List', 'Dictionary', 'IList',
        'IReadOnlyDictionary', 'Action', 'Func', 'Nullable', 'ReadOnlySpan', 'Span',
        'ICollection', 'ISet', 'HashSet', 'Queue', 'Stack', 'KeyValuePair', 'Tuple',
        'CancellationToken', 'CancellationTokenSource', 'TimeSpan', 'DateTime',
        'Type', 'Guid', 'Exception', 'Array', 'IDisposable', 'IComparable',
        'IEquatable', 'IFormattable', 'IConvertible', 'Attribute', 'Delegate',
        'EventHandler', 'AsyncCallback', 'IAsyncResult', 'StringBuilder',
        'MemoryStream', 'Stream', 'TextWriter', 'TextReader', 'Encoding',
        'Regex', 'Match', 'Group', 'Capture',
        'JsonSerializer', 'JsonSerializerOptions', 'JsonSerializerContext',
        'Assembly', 'MethodInfo', 'PropertyInfo', 'FieldInfo',
        # Godot core types
        'Vector2', 'Vector2I', 'Vector3', 'Vector3I', 'Rect2', 'Rect2I',
        'Color', 'StringName', 'Variant', 'Callable', 'NodePath',
        'Transform2D', 'Transform3D', 'Basis', 'Quaternion', 'Aabb',
        'Error', 'Key', 'MouseButton', 'JoyButton', 'JoyAxis',
        # Godot nodes
        'Node', 'Node2D', 'Node3D', 'Control', 'Resource', 'GodotObject',
        'SignalAwaiter', 'Tween', 'RefCounted',
        'InputEvent', 'InputEventKey', 'InputEventMouse', 'InputEventMouseButton',
        'InputEventMouseMotion', 'InputEventJoypadButton', 'InputEventAction',
        'InputEventWithModifiers', 'InputEventPanGesture',
        'Label', 'RichTextLabel', 'Panel', 'Button', 'LineEdit', 'TextEdit',
        'ColorRect', 'AnimatedSprite2D', 'Area2D', 'Camera2D', 'CanvasLayer',
        'GpuParticles2D', 'BackBufferCopy', 'SubViewport', 'Sprite2D',
        'Marker2D', 'FlowContainer', 'HFlowContainer', 'VFlowContainer',
        'Container', 'BoxContainer', 'HBoxContainer', 'VBoxContainer',
        'GridContainer', 'MarginContainer', 'CenterContainer', 'PanelContainer',
        'ScrollContainer', 'TabContainer', 'SplitContainer', 'CanvasGroup',
        'Timer', 'AudioStreamPlayer', 'AudioStreamPlayer2D',
        'PackedScene', 'Shader', 'ShaderMaterial', 'StyleBox', 'Material',
        'Texture2D', 'TextureRect', 'CompressedTexture2D', 'AtlasTexture',
        'AnimationPlayer', 'AnimationTree',
        'ENetConnection', 'ENetPacketPeer', 'ENetMultiplayerPeer',
        'FileAccess', 'DirAccess', 'Image', 'Mesh',
    }

    def _simplify_return_type(self, ret_type: str, extra_known_types: set[str] | None = None) -> str:
        """Simplify return type - keep known types, replace unknown with object?."""
        # Extract the base type name (before ? or <)
        base = ret_type.rstrip('?').replace("[]", "").split('<')[0].split('.')[-1].strip()
        if extra_known_types and base in extra_known_types:
            return ret_type
        if base in self.KNOWN_TYPES:
            return ret_type
        # Keep if it's a known type from scanning (stubs or included source)
        all_known = getattr(self, '_all_stub_type_names', set())
        if base in all_known:
            return ret_type
        # Unknown type - replace with object?
        if ret_type.endswith('?'):
            return "object?"
        return "object"

    def _simplify_params(self, params: str, excluded_type_names: set[str] | None = None,
                         extra_known_types: set[str] | None = None) -> str:
        """Simplify method parameters to avoid dependency chains.
        Replace complex parameter types with object to prevent cascading errors."""
        if not params:
            return ""
        # If params reference problematic content, just return generic params
        if '\n' in params or '\r' in params or len(params) > 300:
            return "params object[] args"
        # Replace unknown types in parameters with object
        all_known = getattr(self, '_all_stub_type_names', set())
        if all_known:
            params = self._replace_unknown_param_types(params, all_known | (extra_known_types or set()))
        return params.strip()

    def _simplify_params_no_type_replace(self, params: str) -> str:
        """Simplify params but do NOT replace types. For override methods."""
        if not params:
            return ""
        if '\n' in params or '\r' in params or len(params) > 300:
            return "params object[] args"
        return params.strip()

    def _replace_unknown_param_types(self, params: str, known_types: set[str]) -> str:
        """Replace unknown type names in parameter lists with 'object'."""
        result = []
        for part in self._split_top_level_commas(params):
            part = part.strip()
            if not part:
                continue
            inner, default_suffix = self._split_param_default(part)
            # Handle 'out', 'ref', 'in', 'params' modifiers
            prefix = ""
            for kw in ('out ', 'ref ', 'in ', 'params '):
                if inner.startswith(kw):
                    prefix = kw
                    inner = inner[len(kw):]
                    break
            # Split into type and name
            # Type names containing < > need special handling
            tokens = inner.rsplit(None, 1)
            if len(tokens) == 2:
                type_part, name_part = tokens
                # Check if the base type is known
                base = type_part.rstrip('?').replace("[]", "").split('<')[0].split('.')[-1].strip()
                if base not in self.KNOWN_TYPES and base not in known_types:
                    nullable = '?' if type_part.endswith('?') else ''
                    type_part = f"object{nullable}"
                result.append(f"{prefix}{type_part} {name_part}{default_suffix}")
            else:
                result.append(f"{prefix}{inner}{default_suffix}")
        return ", ".join(result)

    def _generate_method_body(self, ret_type: str, params: str = "") -> str:
        """Generate a method body based on return type."""
        out_params = []
        for part in self._split_top_level_commas(params):
            stripped = part.strip()
            if stripped.startswith('out '):
                tokens = stripped.split()
                if tokens:
                    out_params.append(tokens[-1].split('=')[0].strip())
        prefix = "".join(f" {{ {name} = default!;" for name in out_params)
        suffix = " }" if out_params else ""
        if ret_type == 'void':
            return f"{prefix or ' {'}{suffix or ' }'}"
        if ret_type == 'bool':
            if out_params:
                return f"{prefix} return false; }}"
            return " => false;"
        if ret_type in ('int', 'float', 'double', 'long', 'ulong', 'short', 'byte', 'uint'):
            return " => 0;"
        if ret_type == 'string':
            return ' => "";'
        if ret_type == 'Task' or ret_type == 'System.Threading.Tasks.Task':
            return " => System.Threading.Tasks.Task.CompletedTask;"
        if ret_type.startswith('Task<') or ret_type.startswith('System.Threading.Tasks.Task<'):
            inner_type = ret_type[ret_type.find('<') + 1:-1].strip()
            return f" => System.Threading.Tasks.Task.FromResult<{inner_type}>(default!)!;"
        if ret_type.endswith('?') and re.match(r'^[A-Z]\w*\??$', ret_type):
            return " => default!;"
        if ret_type.endswith('?'):
            return " => null;"
        return " => default!;"

    def _default_for_type(self, type_name: str) -> str:
        """Get a default value for a type."""
        if type_name == 'string':
            return '""'
        if type_name in ('int', 'float', 'double', 'long', 'ulong', 'short', 'byte', 'uint'):
            return '0'
        if type_name == 'bool':
            return 'false'
        return 'default!'

    def _extract_generic_param_names(self, generic_params: str) -> set[str]:
        if not generic_params:
            return set()
        return {
            token.split(':', 1)[0].strip()
            for token in generic_params.strip()[1:-1].split(',')
            if token.strip()
        }

    def _collect_interface_bases(self, t: TypeInfo) -> list[str]:
        """Collect interface names reachable from a stripped class's original inheritance chain."""
        result: list[str] = []
        visited: set[str] = set()
        queue = list(t.base_types)
        while queue:
            bt = queue.pop(0)
            base_simple = bt.split('<')[0].split('.')[-1].strip()
            if base_simple in visited:
                continue
            visited.add(base_simple)
            if base_simple in self._interface_members:
                result.append(base_simple)
            for type_info in self._types_by_name.values():
                if type_info.name != base_simple:
                    continue
                for parent_bt in type_info.base_types:
                    queue.append(parent_bt)
                break
        return result

    def _collect_interface_members(self, t: TypeInfo, existing_names: set[str]) -> list[TypeMember]:
        """Collect interface members reachable from a stripped class's original inheritance chain."""
        result: list[TypeMember] = []
        visited: set[str] = set()
        queue = list(t.base_types)
        while queue:
            bt = queue.pop(0)
            base_simple = bt.split('<')[0].split('.')[-1].strip()
            if base_simple in visited:
                continue
            visited.add(base_simple)
            if base_simple in self._interface_members:
                for member in self._interface_members[base_simple]:
                    if member.name in existing_names:
                        continue
                    result.append(TypeMember(
                        kind=member.kind,
                        name=member.name,
                        return_type=member.return_type,
                        is_static=False,
                        is_abstract=False,
                        is_virtual=False,
                        is_override=False,
                        params=member.params,
                        generic_params=member.generic_params,
                        default_value='interface',
                    ))
                    existing_names.add(member.name)
            for type_info in self._types_by_name.values():
                if type_info.name != base_simple:
                    continue
                for parent_bt in type_info.base_types:
                    queue.append(parent_bt)
                break
        return result

    def _split_top_level_commas(self, text: str) -> list[str]:
        """Split by commas while respecting generics, tuples, arrays, strings, and defaults."""
        parts: list[str] = []
        depth_angle = 0
        depth_paren = 0
        depth_brace = 0
        depth_bracket = 0
        current: list[str] = []
        in_string = False
        string_char = ''
        escape = False
        for ch in text:
            current.append(ch)
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == string_char:
                    in_string = False
                continue
            if ch in ('"', "'"):
                in_string = True
                string_char = ch
                continue
            if ch == '<':
                depth_angle += 1
                continue
            if ch == '>':
                depth_angle = max(0, depth_angle - 1)
                continue
            if ch == '(':
                depth_paren += 1
                continue
            if ch == ')':
                depth_paren = max(0, depth_paren - 1)
                continue
            if ch == '{':
                depth_brace += 1
                continue
            if ch == '}':
                depth_brace = max(0, depth_brace - 1)
                continue
            if ch == '[':
                depth_bracket += 1
                continue
            if ch == ']':
                depth_bracket = max(0, depth_bracket - 1)
                continue
            if ch == ',' and depth_angle == 0 and depth_paren == 0 and depth_brace == 0 and depth_bracket == 0:
                parts.append("".join(current[:-1]))
                current = []
        if current:
            parts.append("".join(current))
        return parts

    def _split_param_default(self, part: str) -> tuple[str, str]:
        """Split a parameter into declaration and default-value suffix."""
        depth_angle = 0
        depth_paren = 0
        depth_brace = 0
        depth_bracket = 0
        in_string = False
        string_char = ''
        escape = False
        for index, ch in enumerate(part):
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == string_char:
                    in_string = False
                continue
            if ch in ('"', "'"):
                in_string = True
                string_char = ch
                continue
            if ch == '<':
                depth_angle += 1
                continue
            if ch == '>':
                depth_angle = max(0, depth_angle - 1)
                continue
            if ch == '(':
                depth_paren += 1
                continue
            if ch == ')':
                depth_paren = max(0, depth_paren - 1)
                continue
            if ch == '{':
                depth_brace += 1
                continue
            if ch == '}':
                depth_brace = max(0, depth_brace - 1)
                continue
            if ch == '[':
                depth_bracket += 1
                continue
            if ch == ']':
                depth_bracket = max(0, depth_bracket - 1)
                continue
            if ch == '=' and depth_angle == 0 and depth_paren == 0 and depth_brace == 0 and depth_bracket == 0:
                return part[:index].rstrip(), part[index:]
        return part.rstrip(), ""


def _extract_type_names(text: str) -> set[str]:
    """Extract all potential type names from a type expression or parameter list."""
    if not text:
        return set()
    # Find all word tokens that look like type names (start with uppercase)
    names = set()
    for m in re.finditer(r'\b([A-Z]\w+)\b', text):
        names.add(m.group(1))
    return names


# ── Main Entry Point ──

def main(argv: list[str] | None = None) -> int:
    project_root = Path(argv[0] if argv else ".").resolve()
    if not project_root.exists():
        print(f"Project directory does not exist: {project_root}")
        return 1

    print(f"Project root: {project_root}")
    print()

    # Step 1: Scan excluded directories for type definitions
    print("Step 1: Scanning excluded directories for type definitions...")
    scanner = CSharpScanner(project_root)
    excluded_types = scanner.scan_excluded_types()
    print(f"  Found {len(excluded_types)} types in excluded directories")

    # Step 1b: Scan all source for type namespaces and abstract class members
    print("  Scanning all source for type namespaces and abstract class info...")
    scanner.scan_all_type_namespaces()
    print(f"  Found {len(scanner.all_type_namespaces)} unique type names across all source")
    print(f"  Found {len(scanner.abstract_class_members)} abstract classes with abstract members")

    # Step 2: Get simple names for reference scanning
    simple_names = {t.name for t in excluded_types.values()}
    print(f"  Unique simple names: {len(simple_names)}")

    # Step 3: Scan included code for references
    print("\nStep 2: Scanning included code for references to excluded types...")
    referenced_names = scanner.scan_references_in_included_code(simple_names)
    print(f"  Found {len(referenced_names)} directly referenced type names")

    # Step 4: Compute transitive closure - types referenced by stubs themselves
    print("\nStep 3: Computing transitive closure of type dependencies...")
    prev_count = 0
    for iteration in range(10):  # max 10 iterations
        if len(referenced_names) == prev_count:
            break
        prev_count = len(referenced_names)

        # For each referenced type, scan its members for references to other excluded types
        for fqn, type_info in excluded_types.items():
            if type_info.name not in referenced_names:
                continue
            # Check base types
            for bt in type_info.base_types:
                # Extract simple name from "Godot.Node" or "IFoo<Bar>"
                simple = bt.split('<')[0].split('.')[-1].strip()
                if simple in simple_names:
                    referenced_names.add(simple)
            # Check member types
            for member in type_info.members:
                for type_ref in _extract_type_names(member.return_type) | _extract_type_names(member.params):
                    if type_ref in simple_names:
                        referenced_names.add(type_ref)

        print(f"  Iteration {iteration + 1}: {len(referenced_names)} types")

    print(f"  Final referenced types: {len(referenced_names)}")

    # Step 4: Group by namespace
    types_by_ns: dict[str, list[TypeInfo]] = defaultdict(list)
    for fqn, type_info in excluded_types.items():
        if type_info.name in referenced_names:
            types_by_ns[type_info.namespace].append(type_info)

    # Step 5: Generate stub file
    print(f"\nStep 5: Generating stubs...")
    generator = StubCodeGenerator()
    stub_content = generator.generate_stub_file(
        types_by_ns, referenced_names,
        all_type_namespaces=dict(scanner.all_type_namespaces),
        abstract_class_members=scanner.abstract_class_members,
        interface_members=scanner.interface_members,
        included_namespaces=scanner.included_namespaces,
    )

    output_path = project_root / "tools" / "headless-sim" / "HeadlessSim" / "GeneratedStubs.cs"
    output_path.write_text(stub_content, encoding="utf-8")

    # Post-fix: sed-level cleanup
    content = output_path.read_text(encoding="utf-8")
    content = content.replace("??", "?")  # Fix double-nullable
    content = content.replace("params object targets", "params object[] targets")
    # Fix Dictionary ambiguity: remove Godot.Collections using, use fully qualified
    content = content.replace("using Godot.Collections;\n", "")
    # Replace bare Dictionary< with System.Collections.Generic.Dictionary<
    # But not if already fully qualified
    content = re.sub(r'(?<!Generic\.)(?<!\.)\bDictionary<', 'System.Collections.Generic.Dictionary<', content)
    # Fix Array<T> - should be Godot.Collections.Array<T>
    content = re.sub(r'(?<!\.)\bArray<', 'Godot.Collections.Array<', content)
    # Add missing using for addons
    content = content.replace(
        "using Godot;\n",
        "using Godot;\nusing MegaCrit.Sts2.addons.mega_text;\n"
    )
    # NodePool<T>: drop INodePool interface (implementation mismatch)
    content = content.replace(
        "public partial class NodePool<T> : INodePool where T : Node, IPoolable",
        "public partial class NodePool<T> where T : Godot.Node"
    )
    content = content.replace(
        "public static NodePool<T> Init<T>(string scenePath, int prewarmCount) => default!;",
        "public static NodePool<T> Init<T>(string scenePath, int prewarmCount) where T : Godot.Node, IPoolable => default!;"
    )
    # Fix stripped UI screen stubs: keep only the type shell and static factories,
    # not abstract-selection implementations copied from the real UI hierarchy.
    for cls in ['NDeckCardSelectScreen', 'NDeckEnchantSelectScreen',
                'NDeckTransformSelectScreen', 'NDeckUpgradeSelectScreen',
                'NSimpleCardSelectScreen']:
        content = re.sub(
            rf'(public\s+(?:abstract\s+)?partial\s+class\s+{cls}\b)\s*:\s*[^\n]+',
            rf'\1 : Godot.Control, IOverlayScreen, IScreenContext, ICardSelector',
            content
        )
        # Remove selection methods copied from the abstract UI base.
        content = re.sub(
            rf'(class {cls}[^\n]*\n\s*\{{[^}}]*?)public (?:new |override )?[^\n]*GetSelectedCards[^\n]*\n',
            r'\1',
            content
        )
        content = re.sub(
            rf'(class {cls}[^\n]*\n\s*\{{[^}}]*?)public (?:new |override )?[^\n]*GetSelectedCardReward[^\n]*\n',
            r'\1',
            content
        )
        content = re.sub(
            rf'(class {cls}[^\n]*\n\s*\{{[^}}]*?)public [^\n]*GetSelectedCards[^\n]*\n',
            r'\1',
            content
        )
        content = re.sub(
            rf'(class {cls}[^\n]*\n\s*\{{[^}}]*?)public [^\n]*GetSelectedCardReward[^\n]*\n',
            r'\1',
            content
        )
    # Preserve common asset-path signatures that the runtime aggregates heavily.
    content = re.sub(
        r'public static object (AssetPaths) \{ get; set; \}',
        r'public static IEnumerable<string> \1 { get; set; }',
        content
    )
    content = re.sub(
        r'public object (AssetPaths) \{ get; set; \}',
        r'public IEnumerable<string> \1 { get; set; }',
        content
    )
    # Some generated stubs flatten nested enums that callers reference as nested types.
    content = content.replace(
        "public partial class NPlayerHand : Godot.Control\n    {",
        "public partial class NPlayerHand : Godot.Control\n    {\n        public enum Mode { None, Play, SimpleSelect, UpgradeSelect }"
    )
    content = re.sub(r'\n    public enum Mode \{ None, Play, SimpleSelect, UpgradeSelect \}\n', '\n', content, count=1)
    content = content.replace(
        "public partial class NSmokePuffVfx : Godot.Node2D\n    {",
        "public partial class NSmokePuffVfx : Godot.Node2D\n    {\n        public enum SmokePuffColor { Green, Purple }"
    )
    content = re.sub(r'\n    public enum SmokePuffColor \{ Green, Purple \}\n', '\n', content, count=1)
    content = content.replace(
        "public partial class NKaiserCrabBossBackground : Godot.Node\n    {",
        "public partial class NKaiserCrabBossBackground : Godot.Node\n    {\n        public enum ArmSide { Left, Right }"
    )
    content = re.sub(r'\n    public enum ArmSide \{ Left, Right \}\n', '\n', content, count=1)
    content = re.sub(
        r'public object\? (SwooshAwayCompletion) \{ get; set; \}',
        r'public System.Threading.Tasks.TaskCompletionSource? \1 { get; set; }',
        content
    )
    content = content.replace(
        "default(CancellationToken) =>",
        "default(CancellationToken)) =>"
    )
    output_path.write_text(content, encoding="utf-8")
    print(f"  Wrote stubs to: {output_path}")

    # Step 6: Build-error-driven postfix (CS0115/CS0507/CS0546)
    print(f"\nStep 6: Post-fix (build-error-driven)...")
    csproj = str(project_root / "tools" / "headless-sim" / "HeadlessSim" / "HeadlessSim.csproj")
    import subprocess
    prev_error_count = None
    for postfix_round in range(5):
        r = subprocess.run(
            ["dotnet", "build", csproj],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(project_root), timeout=120
        )
        errors = [l for l in r.stdout.split('\n') if 'error CS' in l]
        error_count = len(errors)
        print(f"  Round {postfix_round + 1}: {error_count} errors")

        if error_count == 0:
            print("  *** BUILD SUCCEEDED! ***")
            break
        if prev_error_count is not None and error_count >= prev_error_count:
            print(f"  No improvement. Stopping postfix.")
            break
        prev_error_count = error_count

        content = output_path.read_text(encoding="utf-8")
        lines = content.split('\n')
        changed = 0
        for e in errors:
            if 'GeneratedStubs' not in e:
                continue
            m_line = re.match(r'.*\((\d+),\d+\)', e)
            if not m_line:
                continue
            i = int(m_line.group(1))
            if not (0 < i <= len(lines)):
                continue
            line = lines[i - 1]

            if 'CS0115' in e and ' override ' in line:
                lines[i - 1] = line.replace(' override ', ' new ')
                changed += 1
            elif 'CS0533' in e and ' new ' in line:
                # CS0533: "hides inherited abstract member" → restore override
                lines[i - 1] = line.replace(' new ', ' override ')
                changed += 1
            elif 'CS0534' in e:
                # CS0534: class doesn't implement abstract member → make class abstract
                ms = re.findall(r'[\u201c"](\w+)[\u201d"]', e)
                if ms:
                    cls = ms[0]
                    for j, ln in enumerate(lines):
                        if f'public partial class {cls}' in ln and 'abstract' not in ln:
                            lines[j] = ln.replace(f'public partial class {cls}',
                                                   f'public abstract partial class {cls}')
                            changed += 1
                            break
            elif 'CS0507' in e:
                if 'protected' in e and 'public override' in line:
                    lines[i - 1] = line.replace('public override', 'protected override')
                    changed += 1
                elif 'public' in e and 'protected override' in line:
                    lines[i - 1] = line.replace('protected override', 'public override')
                    changed += 1
            elif 'CS0546' in e and '{ get; set; }' in line:
                lines[i - 1] = line.replace('{ get; set; }', '=> default!;')
                changed += 1

        output_path.write_text('\n'.join(lines), encoding="utf-8")
        print(f"    Fixed {changed} lines")
        if changed == 0:
            break

    # Step 7: Fix remaining CS0534/CS0533 - make problematic subclasses abstract
    # and remove their hiding members
    print(f"\nStep 7: Fixing remaining CS0534/CS0533...")
    content = output_path.read_text(encoding="utf-8")

    # Find classes with CS0534/CS0533 by scanning the stubs file for line numbers in errors
    r = subprocess.run(
        ["dotnet", "build", csproj],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(project_root), timeout=120
    )
    build_output = r.stdout + r.stderr
    cs0534_lines = set()
    for l in build_output.split('\n'):
        if ('CS0534' in l or 'CS0533' in l) and 'GeneratedStubs' in l:
            m = re.match(r'.*\((\d+),', l)
            if m:
                cs0534_lines.add(int(m.group(1)))

    # Find class names from those line numbers
    stub_lines = content.split('\n')
    cs0534_classes = set()
    for line_no in cs0534_lines:
        if 0 < line_no <= len(stub_lines):
            # Search backwards for the class declaration
            for j in range(line_no - 1, max(0, line_no - 30), -1):
                cm = re.search(r'class\s+(\w+)', stub_lines[j])
                if cm:
                    cs0534_classes.add(cm.group(1))
                    break

    if cs0534_classes:
        for cls in cs0534_classes:
            # Make class abstract
            content = re.sub(
                rf'public partial class {cls}\b',
                f'public abstract partial class {cls}',
                content
            )
            # Remove members that hide base abstract members (CS0533)
            # These are non-abstract methods with same name as base abstract methods
            content = re.sub(
                rf'(class {cls}\b[^{{]*\{{)(.*?)(\}})',
                lambda m: m.group(1) + re.sub(
                    r'\n\s*public (?!abstract)[^\n]*(?:GetSelectedCards|GetSelectedCardReward|Process\b)[^\n]*',
                    '', m.group(2)
                ) + m.group(3),
                content,
                flags=re.DOTALL,
                count=1
            )
        output_path.write_text(content, encoding="utf-8")
        print(f"  Made {len(cs0534_classes)} classes abstract: {', '.join(sorted(cs0534_classes))}")

        # Final verification build
        r = subprocess.run(
            ["dotnet", "build", csproj],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(project_root), timeout=120
        )
        final_errors = [l for l in (r.stdout + r.stderr).split('\n') if 'error CS' in l]
        print(f"  After abstract fix: {len(final_errors)} errors")

        # Fix CS0533 (new hides abstract) → change new back to override
        # Re-read build output with raw bytes to avoid encoding issues
        r2 = subprocess.run(
            ["dotnet", "build", csproj],
            capture_output=True, cwd=str(project_root), timeout=120
        )
        build_out2 = r2.stdout.decode('utf-8', errors='replace')
        cs0533_lines = set()
        for l in build_out2.split('\n'):
            if 'CS0533' in l and 'GeneratedStubs' in l:
                m_line = re.match(r'.*\((\d+),\d+\)', l)
                if m_line:
                    cs0533_lines.add(int(m_line.group(1)))

        if cs0533_lines:
            content = output_path.read_text(encoding="utf-8")
            lines = content.split('\n')
            changed = 0
            for i in cs0533_lines:
                if 0 < i <= len(lines):
                    line = lines[i-1]
                    if ' new ' in line:
                        lines[i-1] = line.replace(' new ', ' override ')
                        changed += 1
                    # Note: do NOT add override to plain public methods -
                    # it triggers ~1000 downstream cascading errors
            output_path.write_text('\n'.join(lines), encoding="utf-8")
            print(f"  Fixed {changed} CS0533 (new→override)")

            # Final build
            r3 = subprocess.run(
                ["dotnet", "build", csproj],
                capture_output=True, cwd=str(project_root), timeout=120
            )
            final_out = r3.stdout.decode('utf-8', errors='replace')
            final_errors = [l for l in final_out.split('\n') if 'error CS' in l]
            print(f"  Final errors: {len(final_errors)}")
        else:
            print(f"  No CS0533 found to fix. Final errors: {len(final_errors)}")
    else:
        print("  No CS0534/CS0533 errors found!")

    # Stats
    stub_count = sum(len(ts) for ts in types_by_ns.values())
    print(f"\n  Types scanned: {len(excluded_types)}")
    print(f"  Types referenced: {len(referenced_names)}")
    print(f"  Stubs generated: {stub_count}")

    unreferenced = simple_names - referenced_names
    if unreferenced:
        print(f"  Unreferenced types ({len(unreferenced)}) - no stubs needed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
