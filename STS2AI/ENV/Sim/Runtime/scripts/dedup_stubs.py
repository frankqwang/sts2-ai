#!/usr/bin/env python3
"""
Find and remove duplicate type definitions across stub files.
When a type is defined in both a stub file and the real source (or in two stub files),
comment out the duplicate in the stub file.
"""
import re, os, sys
from collections import defaultdict

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, "..", "..", ".."))

def load_build_errors():
    with open(os.path.join(SCRIPTS_DIR, "build_output.txt"), encoding="utf-8") as f:
        return f.readlines()

def main():
    # First build to get the errors
    build_path = os.path.join(SCRIPTS_DIR, "build_output.txt")
    with open(build_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Extract CS0101 duplicate type errors
    # Format: file(line,col): error CS0101: namespace "X" already contains "TypeName"
    dupes = []
    for l in lines:
        if "error CS0101" not in l:
            continue
        # Extract the file path and type name
        file_match = re.match(r'(.+?)\(\d+,\d+\)', l)
        type_match = re.search(r'[\u201c"](\w+)[\u201d"]', l)
        if file_match and type_match:
            fpath = file_match.group(1).strip()
            type_name = type_match.group(1)
            dupes.append((fpath, type_name))

    # Also extract CS0260 (missing partial modifier) and CS0111 (duplicate member)
    for l in lines:
        if "error CS0260" in l or "error CS0102" in l:
            file_match = re.match(r'(.+?)\(\d+,\d+\)', l)
            type_match = re.search(r'[\u201c"](\w+)[\u201d"]', l)
            if file_match and type_match:
                fpath = file_match.group(1).strip()
                type_name = type_match.group(1)
                dupes.append((fpath, type_name))

    print(f"Found {len(dupes)} duplicate definitions")

    # Group by file
    by_file = defaultdict(set)
    for fpath, type_name in dupes:
        by_file[fpath].add(type_name)

    for fpath, types in sorted(by_file.items()):
        print(f"  {os.path.basename(fpath)}: {', '.join(sorted(types))}")

    # Strategy: for each duplicate, determine which file to keep and which to remove
    # Priority: keep the real source file, remove the stub
    # If both are stubs, keep the one in GodotSharpStub (more complete), remove from Generated*

    stub_dir = os.path.join(ROOT, "tools", "headless-sim", "GodotSharpStub", "src")
    gen_dir = os.path.join(ROOT, "tools", "headless-sim", "HeadlessSim")

    # For types that conflict with re-included real source files,
    # we need to remove them from the stub files
    for fpath, types in by_file.items():
        # Normalize path
        fpath = fpath.replace("\\", "/")

        if "GodotSharpStub" in fpath or "Generated" in fpath:
            # This is a stub file with duplicates - need to comment out the types
            print(f"\nProcessing: {os.path.basename(fpath)}")
            try:
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
            except:
                print(f"  Cannot read {fpath}")
                continue

            original = content
            for type_name in types:
                # Try to comment out the type definition
                # Pattern: public (static|partial)? (class|enum|struct|interface) TypeName ... { ... }
                # This is tricky for multi-line definitions
                # Simple approach for single-line definitions (common in stub files):
                pattern = rf'(public\s+(?:(?:static|partial)\s+)*(?:class|enum|struct|interface)\s+{re.escape(type_name)}\b[^\n]*\{{[^\n]*\}})'
                match = re.search(pattern, content)
                if match:
                    old = match.group(0)
                    content = content.replace(old, f"/* DEDUP: {old} */", 1)
                    print(f"  Commented out single-line: {type_name}")
                else:
                    print(f"  WARNING: Could not auto-remove {type_name} (multi-line?)")

            if content != original:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)

if __name__ == "__main__":
    main()
