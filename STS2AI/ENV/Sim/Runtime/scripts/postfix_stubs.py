#!/usr/bin/env python3
"""
Post-fix GeneratedStubs.cs based on build errors.
- CS0115: remove invalid override declarations
- CS0534: make the class abstract (so it doesn't need to implement)
- CS0246: note for manual fix
"""
import re, os, subprocess
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
STUBS_FILE = ROOT / "tools" / "headless-sim" / "HeadlessSim" / "GeneratedStubs.cs"
CSPROJ = ROOT / "tools" / "headless-sim" / "HeadlessSim" / "HeadlessSim.csproj"

def build():
    result = subprocess.run(
        ["dotnet", "build", str(CSPROJ)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=120
    )
    output = result.stdout + result.stderr
    errors = [l for l in output.split('\n') if 'error CS' in l]
    return errors

def fix_cs0115(errors, content):
    """Remove override declarations that don't match any base class method."""
    lines_to_remove = set()
    for e in errors:
        if 'error CS0115' not in e or 'GeneratedStubs' not in e:
            continue
        m = re.match(r'.*\((\d+),\d+\)', e)
        if m:
            lines_to_remove.add(int(m.group(1)))
    if not lines_to_remove:
        return content
    lines = content.split('\n')
    new_lines = []
    for i, line in enumerate(lines, 1):
        if i in lines_to_remove:
            new_lines.append(f"    // REMOVED(CS0115): {line.strip()}")
        else:
            new_lines.append(line)
    print(f"  CS0115: Commented out {len(lines_to_remove)} invalid override lines")
    return '\n'.join(new_lines)

def fix_cs0534(errors, content):
    """For classes that can't implement abstract members: strip problematic base class,
    keep only Godot base (Control/Node/Node2D) + interfaces."""
    classes_needing_fix = set()
    for e in errors:
        if 'error CS0534' not in e or 'GeneratedStubs' not in e:
            continue
        ms = re.findall(r'[\u201c"](\w+)[\u201d"]', e)
        if ms:
            classes_needing_fix.add(ms[0])
    if not classes_needing_fix:
        return content

    # Known Godot base types to keep
    godot_bases = {'Godot.Control', 'Godot.Node', 'Godot.Node2D', 'Godot.Resource',
                   'Godot.GodotObject', 'Godot.RefCounted', 'Godot.Sprite2D',
                   'Godot.GpuParticles2D', 'Godot.BackBufferCopy', 'Godot.ColorRect'}

    for class_name in classes_needing_fix:
        # Find the class declaration line
        pattern = rf'(public\s+(?:abstract\s+)?partial\s+class\s+{re.escape(class_name)}\b)\s*:\s*([^\n{{]+)'
        m = re.search(pattern, content)
        if not m:
            continue
        decl = m.group(1)
        bases_str = m.group(2).strip()

        # Parse bases, keeping only Godot types and interfaces (start with I + uppercase)
        bases = [b.strip() for b in bases_str.split(',')]
        keep = []
        has_godot_base = False
        for b in bases:
            if b in godot_bases:
                keep.append(b)
                has_godot_base = True
            elif b.startswith('I') and len(b) > 1 and b[1].isupper():
                keep.append(b)  # Keep interfaces

        if not has_godot_base:
            keep.insert(0, 'Godot.Control')

        new_bases = ' : ' + ', '.join(keep) if keep else ''
        content = content[:m.start()] + decl + new_bases + content[m.end():]

    print(f"  CS0534: Stripped abstract base classes from {len(classes_needing_fix)} classes")
    return content

def main():
    max_iterations = 5
    prev_count = None
    backup_content = None

    for iteration in range(max_iterations):
        print(f"\n=== Post-fix iteration {iteration + 1} ===")
        errors = build()
        error_count = len([e for e in errors if 'error CS' in e])
        print(f"  Errors: {error_count}")

        if error_count == 0:
            print("  *** BUILD SUCCEEDED! ***")
            return

        if prev_count is not None and error_count >= prev_count:
            print(f"  No improvement ({prev_count} -> {error_count}). Reverting and stopping.")
            # Revert to backup
            if backup_content:
                STUBS_FILE.write_text(backup_content, encoding='utf-8')
                # Re-count
                errors = build()
                error_count = len([e for e in errors if 'error CS' in e])
                print(f"  Reverted to {error_count} errors")
            break
        prev_count = error_count
        backup_content = STUBS_FILE.read_text(encoding='utf-8')

        content = STUBS_FILE.read_text(encoding='utf-8')
        content = fix_cs0115(errors, content)
        content = fix_cs0534(errors, content)
        STUBS_FILE.write_text(content, encoding='utf-8')

    # Final
    errors = build()
    error_count = len([e for e in errors if 'error CS' in e])
    print(f"\nFinal error count: {error_count}")
    codes = []
    for e in errors:
        m = re.search(r'error (CS\d+)', e)
        if m: codes.append(m.group(1))
    for code, count in Counter(codes).most_common(10):
        print(f"  {code}: {count}")

if __name__ == "__main__":
    main()
