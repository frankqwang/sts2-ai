#!/usr/bin/env python3
"""
Iterative auto-fix: build → parse errors → generate fixes → rebuild.
Handles the most common error patterns automatically.
"""
import re, os, sys, subprocess, json
from collections import defaultdict, Counter

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, "..", "..", ".."))
SRC_DIR = os.path.join(ROOT, "src")
CSPROJ = os.path.join(ROOT, "tools", "headless-sim", "HeadlessSim", "HeadlessSim.csproj")
PATCH_FILE = os.path.join(ROOT, "tools", "headless-sim", "HeadlessSim", "AutoFixPatches.cs")

def build():
    """Run dotnet build and return (error_count, error_lines)."""
    result = subprocess.run(
        ["dotnet", "build", CSPROJ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=ROOT, timeout=120
    )
    output = result.stdout + result.stderr
    errors = [l for l in output.split('\n') if 'error CS' in l]

    # Extract error count from summary line
    count_match = re.search(r'(\d+)\s*个错误', output)
    if not count_match:
        count_match = re.search(r'(\d+)\s*Error', output)
    count = int(count_match.group(1)) if count_match else len(errors)

    return count, errors

def parse_errors(error_lines):
    """Parse error lines into structured data."""
    parsed = []
    for l in error_lines:
        m = re.match(r'(.+?)\((\d+),(\d+)\):\s*error\s+(CS\d+):\s*(.*?)(?:\s*\[|$)', l)
        if m:
            parsed.append({
                'file': m.group(1).strip().replace('\\', '/'),
                'line': int(m.group(2)),
                'col': int(m.group(3)),
                'code': m.group(4),
                'msg': m.group(5).strip(),
            })
    return parsed

def find_member_signature(type_name, member_name):
    """Find a member's signature in source code."""
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
            if not re.search(rf'\b(class|struct)\s+{re.escape(type_name)}\b', content):
                continue
            # Property
            pm = re.search(rf'public\s+(?:(?:static|override|virtual|new|readonly)\s+)*(\S+)\s+{re.escape(member_name)}\s*\{{', content)
            if pm:
                return 'property', pm.group(1), 'static' in pm.group(0)
            # Method
            mm = re.search(rf'public\s+(?:(?:static|override|virtual|new|async)\s+)*(\S+)\s+{re.escape(member_name)}\s*\(', content)
            if mm:
                return 'method', mm.group(1), 'static' in mm.group(0)
            # Field
            fm = re.search(rf'public\s+(?:(?:static|readonly|const)\s+)*(\S+)\s+{re.escape(member_name)}\s*[;=]', content)
            if fm:
                return 'field', fm.group(1), 'static' in fm.group(0)
    return None

def find_type_namespace(type_name):
    """Find the namespace of a type."""
    # Check all stub files and source
    search_dirs = [
        os.path.join(ROOT, "tools", "headless-sim", "GodotSharpStub", "src"),
        os.path.join(ROOT, "tools", "headless-sim", "HeadlessSim"),
        SRC_DIR,
    ]
    for search_dir in search_dirs:
        for dirpath, _, filenames in os.walk(search_dir):
            for fname in filenames:
                if not fname.endswith('.cs') or fname.endswith('.uid'):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, encoding='utf-8') as f:
                        content = f.read()
                except:
                    continue
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
    return None

def generate_fixes(parsed_errors):
    """Generate C# fix code from parsed errors."""
    # Group by error type
    by_code = defaultdict(list)
    for e in parsed_errors:
        by_code[e['code']].append(e)

    fixes = defaultdict(set)  # namespace -> set of member declarations

    # CS1061: 'Type' does not contain 'Member'
    for e in by_code.get('CS1061', []):
        ms = re.findall(r'[\u201c"](\w+)[\u201d"]', e['msg'])
        if len(ms) >= 2:
            type_name, member_name = ms[0], ms[1]
            if type_name in ('object', 'dynamic', 'void', 'int', 'string', 'bool', 'Dispose'):
                continue
            ns = find_type_namespace(type_name)
            if ns:
                sig = find_member_signature(type_name, member_name)
                if sig and sig[0] == 'property':
                    fixes[ns].add(f"    public partial class {type_name} {{ public dynamic {member_name} {{ get; set; }} }}")
                else:
                    fixes[ns].add(f"    public partial class {type_name} {{ public dynamic {member_name}(dynamic a = null, dynamic b = null, dynamic c = null, dynamic d = null) => null; }}")

    # CS0103: name does not exist in current context
    for e in by_code.get('CS0103', []):
        ms = re.findall(r'[\u201c"](\w+)[\u201d"]', e['msg'])
        if ms:
            name = ms[0]
            ns = find_type_namespace(name)
            if not ns:
                # Might be a static class reference - try to find and create
                ns_from_src = None
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
                        if re.search(rf'\b(class|enum|struct)\s+{re.escape(name)}\b', content):
                            ns_match = re.search(r'namespace\s+([\w.]+)', content)
                            if ns_match:
                                ns_from_src = ns_match.group(1)
                                kind_match = re.search(rf'(class|enum|struct)\s+{re.escape(name)}', content)
                                kind = kind_match.group(1) if kind_match else 'class'
                                is_static = bool(re.search(rf'static\s+{kind}\s+{re.escape(name)}', content))
                                if kind == 'enum':
                                    fixes[ns_from_src].add(f"    public enum {name} {{ None }}")
                                elif is_static:
                                    fixes[ns_from_src].add(f"    public static partial class {name} {{ }}")
                                else:
                                    fixes[ns_from_src].add(f"    public partial class {name} : Godot.Node {{ }}")
                                break
                    if ns_from_src:
                        break

    # CS0117: 'Type' does not contain 'StaticMember'
    for e in by_code.get('CS0117', []):
        ms = re.findall(r'[\u201c"]([^\u201d"]+)[\u201d"]', e['msg'])
        if len(ms) >= 2:
            type_name, member_name = ms[0], ms[1]
            ns = find_type_namespace(type_name)
            if ns:
                fixes[ns].add(f"    public partial class {type_name} {{ public static dynamic {member_name} {{ get; set; }} }}")

    # CS1739: named parameter doesn't exist - add it by making method accept dynamic params
    # (already handled by dynamic params pattern)

    # CS1729: constructor argument count mismatch
    for e in by_code.get('CS1729', []):
        ms = re.findall(r'[\u201c"](\w+)[\u201d"]', e['msg'])
        arg_match = re.search(r'(\d+)\s*个参数', e['msg'])
        if ms and arg_match:
            type_name = ms[0]
            arg_count = int(arg_match.group(1))
            ns = find_type_namespace(type_name)
            if ns:
                params = ", ".join([f"dynamic a{i} = null" for i in range(arg_count)])
                fixes[ns].add(f"    public partial class {type_name} {{ public {type_name}({params}) {{ }} }}")

    return fixes

def write_patch_file(fixes):
    """Write the accumulated fixes to the patch file."""
    lines = [
        "// AUTO-GENERATED PATCHES by auto_fix_loop.py",
        "// Fixes for compilation errors in stub compatibility layer",
        "#nullable enable",
        "#pragma warning disable CS0414, CS0649, CS0108, CS0114, CS0109, CS8618",
        "",
    ]
    for ns in sorted(fixes.keys()):
        lines.append(f"namespace {ns}")
        lines.append("{")
        for fix in sorted(fixes[ns]):
            lines.append(fix)
        lines.append("}")
        lines.append("")

    with open(PATCH_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return len([f for s in fixes.values() for f in s])

def main():
    max_iterations = 5
    prev_count = None

    for i in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"=== Iteration {i+1} ===")
        print(f"{'='*60}")

        count, error_lines = build()
        print(f"Errors: {count}")

        if count == 0:
            print("\n*** BUILD SUCCEEDED! ***")
            return

        if prev_count is not None and count >= prev_count:
            print(f"No improvement (was {prev_count}, now {count}). Stopping.")
            break

        prev_count = count

        parsed = parse_errors(error_lines)
        print(f"Parsed {len(parsed)} error entries")

        # Show error distribution
        code_counts = Counter(e['code'] for e in parsed)
        print("Error distribution:")
        for code, cnt in code_counts.most_common(10):
            print(f"  {code}: {cnt}")

        # Load existing patches
        existing_fixes = defaultdict(set)
        if os.path.exists(PATCH_FILE):
            with open(PATCH_FILE, encoding='utf-8') as f:
                content = f.read()
            # Simple parse: find namespace blocks and their contents
            for m in re.finditer(r'namespace\s+([\w.]+)\s*\{([^}]*)\}', content, re.DOTALL):
                ns = m.group(1)
                for line in m.group(2).strip().split('\n'):
                    line = line.strip()
                    if line:
                        existing_fixes[ns].add(line)

        # Generate new fixes
        new_fixes = generate_fixes(parsed)

        # Merge
        for ns, members in new_fixes.items():
            existing_fixes[ns] |= members

        fix_count = write_patch_file(existing_fixes)
        print(f"Wrote {fix_count} fixes to AutoFixPatches.cs")

    # Final build
    print(f"\n{'='*60}")
    print("=== Final build ===")
    print(f"{'='*60}")
    count, _ = build()
    print(f"Final error count: {count}")

if __name__ == "__main__":
    main()
