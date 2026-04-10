#!/usr/bin/env python3
"""
Analyze CS0246 errors in GeneratedStubs.cs to find what using directives are needed.
For each missing type, find its namespace in the source code.
"""
import re, os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC = ROOT / "src"
STUBS_DIR = ROOT / "tools" / "headless-sim"

def find_namespace_for_type(type_name: str) -> str | None:
    """Search all .cs files to find the namespace of a type."""
    for dirpath, _, filenames in os.walk(SRC):
        for fname in filenames:
            if not fname.endswith('.cs') or fname.endswith('.uid'):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, encoding='utf-8-sig') as f:
                    content = f.read()
            except:
                continue
            if re.search(rf'\b(class|enum|struct|interface)\s+{re.escape(type_name)}\b', content):
                ns = None
                for m in re.finditer(r'namespace\s+([\w.]+)', content):
                    ns = m.group(1)
                if ns:
                    return ns
    # Also check GodotSharpStub
    for cs_file in (STUBS_DIR / "GodotSharpStub" / "src").rglob("*.cs"):
        try:
            content = cs_file.read_text(encoding='utf-8-sig')
        except:
            continue
        if re.search(rf'\b(class|enum|struct|interface)\s+{re.escape(type_name)}\b', content):
            for m in re.finditer(r'namespace\s+([\w.]+)', content):
                return m.group(1)
    return None

def main():
    build_output = ROOT / "tools" / "headless-sim" / "scripts" / "build_output.txt"
    with open(build_output, encoding='utf-8') as f:
        lines = f.readlines()

    # CS0246 in GeneratedStubs.cs
    cs246 = [l for l in lines if 'error CS0246' in l and 'GeneratedStubs' in l]
    missing = []
    for l in cs246:
        m = re.search(r'[\u201c"](\w+)[\u201d"]', l)
        if m:
            missing.append(m.group(1))

    counts = Counter(missing)
    unique_missing = sorted(counts.keys())
    print(f"Unique missing types in GeneratedStubs.cs: {len(unique_missing)}")

    # Find namespaces
    ns_needed = defaultdict(list)
    not_found = []
    for name in unique_missing:
        ns = find_namespace_for_type(name)
        if ns:
            ns_needed[ns].append(name)
        else:
            not_found.append(name)

    print(f"\nNamespaces needed ({len(ns_needed)}):")
    for ns in sorted(ns_needed.keys()):
        types = ns_needed[ns]
        print(f"  using {ns};  // {', '.join(types[:5])}{'...' if len(types)>5 else ''}")

    if not_found:
        print(f"\nTypes not found in source ({len(not_found)}):")
        for name in not_found:
            print(f"  {name} ({counts[name]} refs)")

    # CS0534 abstract member errors
    cs534 = [l for l in lines if 'error CS0534' in l]
    abstract_classes = set()
    for l in cs534:
        ms = re.findall(r'[\u201c"](\w+)[\u201d"]', l)
        if ms:
            abstract_classes.add(ms[0])
    if abstract_classes:
        print(f"\nClasses missing abstract implementations ({len(abstract_classes)}):")
        for name in sorted(abstract_classes):
            print(f"  {name}")

if __name__ == "__main__":
    main()
