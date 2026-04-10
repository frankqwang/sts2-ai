#!/usr/bin/env python3
"""Generate csproj Compile Remove entries for all files with errors."""
import re, os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, "..", "..", ".."))

with open(os.path.join(SCRIPTS_DIR, "build_output.txt"), encoding="utf-8") as f:
    lines = f.readlines()

error_files = set()
for l in lines:
    if "error CS" not in l:
        continue
    m = re.match(r'(.+?)\(\d+,\d+\)', l)
    if m:
        fpath = m.group(1).strip().replace("\\", "/")
        if "tools/headless-sim" not in fpath:
            error_files.add(fpath)

# Convert to relative paths from csproj
root_norm = ROOT.replace("\\", "/")
excludes = []
for f in sorted(error_files):
    rel = f.replace(root_norm + "/", "")
    csproj_rel = "../../../" + rel
    excludes.append(csproj_rel)

print(f"Total files to exclude: {len(excludes)}")
print()
print("    <!-- Exclude files with stub-related compilation errors -->")
for e in excludes:
    print(f'    <Compile Remove="{e}" />')
