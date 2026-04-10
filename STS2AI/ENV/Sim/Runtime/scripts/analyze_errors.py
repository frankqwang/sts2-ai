#!/usr/bin/env python3
"""Analyze build errors from HeadlessSim compilation."""
import re, os, sys
from collections import Counter

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "build_output.txt")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    def extract_names(error_code):
        filtered = [l for l in lines if f"error {error_code}" in l]
        names = []
        for l in filtered:
            m = re.search(r'[\u201c"](\w+)[\u201d"]', l)
            if m:
                names.append(m.group(1))
        return Counter(names)

    def extract_full_names(error_code):
        filtered = [l for l in lines if f"error {error_code}" in l]
        names = []
        for l in filtered:
            m = re.search(r'[\u201c"]([^"\u201d]+)[\u201d"]', l)
            if m:
                names.append(m.group(1))
        return Counter(names)

    print("=== CS0246 Remaining Missing Types ===")
    for name, count in extract_names("CS0246").most_common(50):
        print(f"  {count:4d}  {name}")

    print("\n=== CS0103 Remaining Missing Names ===")
    for name, count in extract_names("CS0103").most_common(50):
        print(f"  {count:4d}  {name}")

    print("\n=== CS0119 Top Issues ===")
    for name, count in extract_full_names("CS0119").most_common(20):
        print(f"  {count:4d}  {name}")

    print("\n=== CS1061 Missing Members ===")
    for name, count in extract_names("CS1061").most_common(20):
        print(f"  {count:4d}  {name}")

    print("\n=== CS0117 Missing Static Members ===")
    for name, count in extract_full_names("CS0117").most_common(20):
        print(f"  {count:4d}  {name}")

if __name__ == "__main__":
    main()
