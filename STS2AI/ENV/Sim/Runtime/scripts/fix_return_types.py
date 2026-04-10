#!/usr/bin/env python3
"""
Fix property return types in stub files.
Reads the real source to find the actual return type of properties
that are currently typed as object?.
"""
import re, os, glob

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, "..", "..", ".."))
SRC_DIR = os.path.join(ROOT, "src")

def find_property_type(class_name, prop_name):
    """Find the actual return type of a property in the real source."""
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
            if not re.search(rf'\b(class|struct)\s+{re.escape(class_name)}\b', content):
                continue
            # Look for property definition
            m = re.search(
                rf'public\s+(?:(?:static|override|virtual|new)\s+)*(\S+)\s+{re.escape(prop_name)}\s*\{{',
                content
            )
            if m:
                return m.group(1)
    return None

def main():
    # Key properties from error analysis - these return object? but should return specific types
    # Format: (class_name, property_name)
    key_props = [
        ("NRun", "GlobalUi"),
        ("NRun", "RunMusicController"),
        ("NRun", "EventRoom"),
        ("NRun", "MerchantRoom"),
        ("NRun", "TreasureRoom"),
        ("NCombatRoom", "CombatVfxContainer"),
        ("NCombatRoom", "BackCombatVfxContainer"),
        ("NCombatRoom", "Ui"),
        ("NCombatRoom", "Background"),
        ("NCombatRoom", "CreatureNodes"),
        ("NCreature", "SpineController"),
        ("NCreature", "Visuals"),
        ("NCreature", "OrbManager"),
        ("NCreature", "Entity"),
        ("NCreature", "Hitbox"),
        ("NCreature", "Body"),
        ("NCreatureVisuals", "SpineBody"),
        ("NGame", "CurrentRunNode"),
        ("NPlayerHand", "ActiveHolders"),
        ("NPlayerHand", "CurrentMode"),
        ("NPlayerHand", "PeekButton"),
        ("NSelectedHandCardContainer", "Holders"),
        ("NHandCardHolder", "CardNode"),
        ("NCardHolder", "CardModel"),
        ("NCardHolder", "CardNode"),
        ("NRestSiteCharacter", "Hitbox"),
        ("NTreasureRoom", "ProceedButton"),
        ("NMapScreen", "Visible"),
        ("MegaSprite", "GetSkeleton"),
        ("MegaSprite", "GetAnimationState"),
        ("NRestSiteRoom", "ProceedButton"),
        ("NRestSiteRoom", "Characters"),
        ("NRestSiteRoom", "Options"),
        ("NMerchantRoom", "ProceedButton"),
        ("NMerchantRoom", "Inventory"),
        ("NEventRoom", "VfxContainer"),
    ]

    results = []
    for cls, prop in key_props:
        real_type = find_property_type(cls, prop)
        results.append((cls, prop, real_type))
        status = real_type or "NOT FOUND"
        print(f"  {cls}.{prop} -> {status}")

    # Now fix the stub files
    stub_files = (
        glob.glob(os.path.join(ROOT, "tools", "headless-sim", "GodotSharpStub", "src", "*.cs")) +
        glob.glob(os.path.join(ROOT, "tools", "headless-sim", "HeadlessSim", "Generated*.cs"))
    )

    total_fixes = 0
    for fpath in stub_files:
        try:
            with open(fpath, encoding='utf-8') as f:
                content = f.read()
        except:
            continue

        original = content
        for cls, prop, real_type in results:
            if not real_type:
                continue
            # Map Godot types
            type_map = {
                "Control": "Godot.Control",
                "Node": "Godot.Node",
                "Node2D": "Godot.Node2D",
                "TextureRect": "Godot.TextureRect",
            }
            mapped_type = type_map.get(real_type, real_type)
            # If type is a game type, check if it has a Godot prefix needed
            if not mapped_type.startswith("Godot.") and not mapped_type.startswith("System.") and mapped_type[0].isupper():
                # Keep as-is - it's a game type that should be in scope
                pass

            # Replace: public object? PropName { get; set; } -> public MappedType? PropName { get; set; }
            # But only within the right class context
            # Simple approach: replace in any class that matches
            old = f"public object? {prop} {{ get; set; }}"
            # For nullable ref types
            new = f"public {mapped_type}? {prop} {{ get; set; }}"
            if old in content:
                content = content.replace(old, new, 1)  # Only first occurrence to be safe

        if content != original:
            fixes = sum(1 for a, b in zip(original, content) if a != b)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(content)
            fname = os.path.basename(fpath)
            print(f"  Updated {fname}")
            total_fixes += 1

    print(f"\nTotal files updated: {total_fixes}")

if __name__ == "__main__":
    main()
