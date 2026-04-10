#!/usr/bin/env python3
import re, os
from collections import Counter

base = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base, "build_output.txt"), encoding="utf-8") as f:
    lines = f.readlines()

# CS1061: member not found on type
cs1061 = [l for l in lines if "error CS1061" in l]
pairs = []
for l in cs1061:
    ms = re.findall(r'[\u201c"](\w+)[\u201d"]', l)
    if len(ms) >= 2:
        pairs.append((ms[0], ms[1]))
c = Counter(pairs)
print("=== CS1061 Missing Members (type.member) top 40 ===")
for (t, m), count in c.most_common(40):
    print(f"  {count:4d}  {t}.{m}")

# CS1955: non-invocable
cs1955 = [l for l in lines if "error CS1955" in l]
names = []
for l in cs1955:
    m = re.search(r'[\u201c"](\w+)[\u201d"]', l)
    if m: names.append(m.group(1))
c2 = Counter(names)
print()
print("=== CS1955 Non-invocable top 20 ===")
for name, count in c2.most_common(20):
    print(f"  {count:4d}  {name}")

# CS0117: static member missing
cs117 = [l for l in lines if "error CS0117" in l]
pairs117 = []
for l in cs117:
    ms = re.findall(r'[\u201c"]([^\u201d"]+)[\u201d"]', l)
    if len(ms) >= 2:
        pairs117.append((ms[0], ms[1]))
c3 = Counter(pairs117)
print()
print("=== CS0117 Missing Static Members top 20 ===")
for (t, m), count in c3.most_common(20):
    print(f"  {count:4d}  {t}.{m}")

# CS0103: name doesn't exist
cs103 = [l for l in lines if "error CS0103" in l]
names103 = []
for l in cs103:
    m = re.search(r'[\u201c"](\w+)[\u201d"]', l)
    if m: names103.append(m.group(1))
c4 = Counter(names103)
print()
print("=== CS0103 Missing Names top 20 ===")
for name, count in c4.most_common(20):
    print(f"  {count:4d}  {name}")
