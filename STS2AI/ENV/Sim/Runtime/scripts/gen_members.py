#!/usr/bin/env python3
"""
Generate missing members for stub classes based on build errors.
Parses CS1061/CS0117/CS0103 errors, finds the real member signatures
in source, and generates stubs.
"""
import re, os, sys
from collections import Counter, defaultdict

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, "..", "..", ".."))
SRC_DIR = os.path.join(ROOT, "src")

def load_build_errors():
    with open(os.path.join(SCRIPTS_DIR, "build_output.txt"), encoding="utf-8") as f:
        return f.readlines()

def find_member_in_source(type_name, member_name):
    """Search source for a member definition in the given type."""
    for dirpath, _, filenames in os.walk(SRC_DIR):
        for fname in filenames:
            if not fname.endswith('.cs') or fname.endswith('.uid'):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, encoding='utf-8') as f:
                    content = f.read()
            except:
                continue

            # Check if this file defines the type
            if not re.search(rf'\b(class|struct|interface)\s+{re.escape(type_name)}\b', content):
                continue

            # Look for the member
            # Property pattern
            prop_match = re.search(
                rf'public\s+(?:(?:static|override|virtual|new|readonly)\s+)*(\S+)\s+{re.escape(member_name)}\s*{{',
                content
            )
            if prop_match:
                ret_type = prop_match.group(1)
                is_static = 'static' in prop_match.group(0)
                return ('property', ret_type, is_static)

            # Method pattern
            method_match = re.search(
                rf'public\s+(?:(?:static|override|virtual|new|async)\s+)*(\S+)\s+{re.escape(member_name)}\s*\(',
                content
            )
            if method_match:
                ret_type = method_match.group(1)
                is_static = 'static' in method_match.group(0)
                # Get parameters
                start = method_match.end() - 1
                depth = 0
                end = start
                for i in range(start, min(start + 500, len(content))):
                    if content[i] == '(':
                        depth += 1
                    elif content[i] == ')':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                params = content[start+1:end-1].strip()
                return ('method', ret_type, is_static, params)

            # Field pattern
            field_match = re.search(
                rf'public\s+(?:(?:static|readonly|const)\s+)*(\S+)\s+{re.escape(member_name)}\s*[;=]',
                content
            )
            if field_match:
                ret_type = field_match.group(1)
                is_static = 'static' in field_match.group(0)
                is_const = 'const' in field_match.group(0)
                return ('field', ret_type, is_static, is_const)

            # Enum member (for nested enums/classes)
            enum_match = re.search(
                rf'public\s+enum\s+{re.escape(member_name)}\b',
                content
            )
            if enum_match:
                return ('enum', member_name, True)

    return None

def simplify_type(t):
    """Simplify a type for stub generation."""
    # Map common types
    mapping = {
        'Task': 'System.Threading.Tasks.Task',
        'bool': 'bool',
        'int': 'int',
        'float': 'float',
        'double': 'double',
        'string': 'string',
        'void': 'void',
    }
    if t in mapping:
        return mapping[t]
    return t

def default_value(ret_type):
    """Generate a default return value for a type."""
    if ret_type in ('void',):
        return None
    if ret_type in ('bool',):
        return 'false'
    if ret_type in ('int', 'float', 'double', 'long', 'ulong', 'short', 'byte'):
        return '0'
    if ret_type == 'string':
        return '""'
    if ret_type.startswith('Task'):
        return 'System.Threading.Tasks.Task.CompletedTask'
    return 'default!'

def generate_member_stub(member_name, info):
    """Generate a C# member stub from source analysis."""
    if info is None:
        # Fallback: generate as method with object? params
        return f"        public void {member_name}(object? a = null, object? b = null, object? c = null, object? d = null) {{ }}"

    kind = info[0]
    if kind == 'property':
        ret_type, is_static = info[1], info[2]
        static = "static " if is_static else ""
        return f"        public {static}object? {member_name} {{ get; set; }}"

    elif kind == 'method':
        ret_type, is_static, params = info[1], info[2], info[3]
        static = "static " if is_static else ""
        # Simplify: use object? params
        dv = default_value(ret_type)
        if ret_type == 'void':
            return f"        public {static}void {member_name}(object? a = null, object? b = null, object? c = null, object? d = null) {{ }}"
        elif ret_type == 'Task' or ret_type.startswith('Task'):
            return f"        public {static}System.Threading.Tasks.Task {member_name}(object? a = null, object? b = null, object? c = null, object? d = null) => System.Threading.Tasks.Task.CompletedTask;"
        elif ret_type == 'bool':
            return f"        public {static}bool {member_name}(object? a = null, object? b = null, object? c = null, object? d = null) => false;"
        elif ret_type == 'string':
            return f"        public {static}string {member_name}(object? a = null, object? b = null, object? c = null, object? d = null) => \"\";"
        else:
            return f"        public {static}object? {member_name}(object? a = null, object? b = null, object? c = null, object? d = null) => null;"

    elif kind == 'field':
        ret_type, is_static, is_const = info[1], info[2], info[3]
        static = "static " if is_static else ""
        if is_const:
            if ret_type == 'string':
                return f"        public const string {member_name} = \"\";"
            else:
                return f"        public const int {member_name} = 0;"
        return f"        public {static}object? {member_name} {{ get; set; }}"

    elif kind == 'enum':
        return f"        public enum {member_name} {{ None }}"

    return f"        // TODO: {member_name}"

def main():
    lines = load_build_errors()

    # Collect CS1061: type.member pairs
    cs1061 = [l for l in lines if "error CS1061" in l]
    needed_members = defaultdict(set)  # type -> set of members
    for l in cs1061:
        ms = re.findall(r'[\u201c"](\w+)[\u201d"]', l)
        if len(ms) >= 2:
            type_name, member_name = ms[0], ms[1]
            if type_name != 'object':  # Skip object - those are from unresolved chains
                needed_members[type_name].add(member_name)

    # Collect CS0117: Type.StaticMember
    cs117 = [l for l in lines if "error CS0117" in l]
    for l in cs117:
        ms = re.findall(r'[\u201c"]([^\u201d"]+)[\u201d"]', l)
        if len(ms) >= 2:
            type_name, member_name = ms[0], ms[1]
            needed_members[type_name].add(member_name)

    print(f"Types needing members: {len(needed_members)}")
    total = sum(len(v) for v in needed_members.values())
    print(f"Total members needed: {total}")

    # For each type.member, find the real signature
    stubs_by_type = defaultdict(list)
    found = 0
    not_found = 0
    for type_name in sorted(needed_members.keys()):
        for member_name in sorted(needed_members[type_name]):
            info = find_member_in_source(type_name, member_name)
            stub = generate_member_stub(member_name, info)
            stubs_by_type[type_name].append(stub)
            if info:
                found += 1
            else:
                not_found += 1

    print(f"Found in source: {found}")
    print(f"Not found (using fallback): {not_found}")

    # Now we need to inject these into the existing stub files or the generated stubs
    # Strategy: write a new partial class file
    output_lines = [
        "// AUTO-GENERATED MEMBER STUBS",
        "// Generated by gen_members.py - adds missing members to stub classes",
        "#nullable enable",
        "#pragma warning disable CS0414, CS0649, CS0108, CS0114, CS0109",
        "",
    ]

    # Find namespace for each type
    for type_name in sorted(stubs_by_type.keys()):
        ns = find_type_namespace(type_name)
        if not ns:
            print(f"  WARNING: No namespace found for {type_name}")
            continue

        members = stubs_by_type[type_name]
        output_lines.append(f"namespace {ns}")
        output_lines.append("{")
        output_lines.append(f"    public partial class {type_name}")
        output_lines.append("    {")
        for m in members:
            output_lines.append(m)
        output_lines.append("    }")
        output_lines.append("}")
        output_lines.append("")

    output_path = os.path.join(ROOT, "tools", "headless-sim", "HeadlessSim", "GeneratedMemberStubs.cs")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"\nWrote to: {output_path}")


def find_type_namespace(type_name):
    """Find the namespace of a type by searching all stub files and source."""
    # Check stub files first
    stub_dir = os.path.join(ROOT, "tools", "headless-sim", "GodotSharpStub", "src")
    for d in [stub_dir]:
        for fname in os.listdir(d):
            if not fname.endswith('.cs'):
                continue
            fpath = os.path.join(d, fname)
            try:
                with open(fpath, encoding='utf-8') as f:
                    content = f.read()
            except:
                continue
            # Find the type and its enclosing namespace
            type_pos = -1
            for m in re.finditer(rf'\b(class|struct|interface|enum)\s+{re.escape(type_name)}\b', content):
                type_pos = m.start()
                break
            if type_pos >= 0:
                # Find the most recent namespace before this position
                ns = None
                for m in re.finditer(r'namespace\s+([\w.]+)', content[:type_pos]):
                    ns = m.group(1)
                if ns:
                    return ns

    # Check generated stubs
    gen_path = os.path.join(ROOT, "tools", "headless-sim", "HeadlessSim", "GeneratedMissingStubs.cs")
    if os.path.exists(gen_path):
        with open(gen_path, encoding='utf-8') as f:
            content = f.read()
        type_pos = -1
        for m in re.finditer(rf'\b(class|struct|interface|enum)\s+{re.escape(type_name)}\b', content):
            type_pos = m.start()
            break
        if type_pos >= 0:
            ns = None
            for m in re.finditer(r'namespace\s+([\w.]+)', content[:type_pos]):
                ns = m.group(1)
            if ns:
                return ns

    # Check source
    for dirpath, _, filenames in os.walk(SRC_DIR):
        for fname in filenames:
            if not fname.endswith('.cs') or fname.endswith('.uid'):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, encoding='utf-8') as f:
                    content = f.read()
            except:
                continue
            if re.search(rf'\b(class|struct|interface|enum)\s+{re.escape(type_name)}\b', content):
                ns_match = re.search(r'namespace\s+([\w.]+)', content)
                if ns_match:
                    return ns_match.group(1)

    # Special cases
    specials = {
        'Tween': 'Godot',
        'LobbyCreated_t': 'Steamworks',
        'LobbyEnter_t': 'Steamworks',
        'SteamNetworkingConfigValue_t': 'Steamworks',
        'SteamNetworkingMessage_t': 'Steamworks',
        'ELobbyType': 'Steamworks',
        'EChatRoomEnterResponse': 'Steamworks',
        'ESteamNetworkingConnectionState': 'Steamworks',
        'ESteamNetworkingConfigValue': 'Steamworks',
        'ESteamNetworkingConfigDataType': 'Steamworks',
        'NetError': 'Steamworks',
        'SteamNetworkingSockets': 'Steamworks',
        'SteamMatchmaking': 'Steamworks',
        'SteamRemoteStorage': 'Steamworks',
        'SteamUser': 'Steamworks',
        'SteamUGC': 'Steamworks',
        'SteamFriends': 'Steamworks',
        'SteamUtils': 'Steamworks',
    }
    return specials.get(type_name)


if __name__ == "__main__":
    main()
