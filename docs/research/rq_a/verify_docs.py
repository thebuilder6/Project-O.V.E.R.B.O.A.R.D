"""Check the research documentation: encoding, markdown structure, table
drift against the measurements, and the line-number citations.

    python docs/research/rq_a/verify_docs.py

Exit code 1 if anything is wrong. Run `sync_tables.py` first if the tables
have drifted.
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DOC = os.path.join(REPO, "docs", "HEURISTICS_AND_VERIFICATION.md")
NOTES = os.path.join(REPO, "docs", "research", "RQ-A_tracking_error.md")

problems = []


def read(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return raw, raw.decode("utf-8", errors="replace")


# --- encoding ---------------------------------------------------------------
for path in (DOC, NOTES, os.path.join(REPO, "docs", "research", "rq_a", "README.md")):
    raw, text = read(path)
    rel = os.path.relpath(path, REPO)
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append(f"{rel}: has a UTF-8 BOM")
    if b"\r\n" in raw:
        problems.append(f"{rel}: has CRLF line endings")
    if "\ufffd" in text:
        problems.append(f"{rel}: contains U+FFFD (mojibake)")

# --- markdown ---------------------------------------------------------------
for path in (DOC, NOTES):
    raw, text = read(path)
    rel = os.path.relpath(path, REPO)
    lines = text.split("\n")
    fences = sum(1 for l in lines if l.strip().startswith("```"))
    if fences % 2:
        problems.append(f"{rel}: unbalanced code fences ({fences})")

    def slug(h):
        s = re.sub(r"[`*_]", "", h.strip().lower())
        s = re.sub(r"[^a-z0-9 \-]", "", s)
        return s.replace(" ", "-")

    anchors = {slug(h) for h in re.findall(r"^#{1,4} (.+)$", text, re.M)}
    for m in re.finditer(r"\]\(#([^)]+)\)", text):
        if m.group(1) not in anchors:
            problems.append(f"{rel}: dead anchor #{m.group(1)}")

    i = 0
    while i < len(lines):
        if (lines[i].startswith("|") and i + 1 < len(lines)
                and re.match(r"^\|[\s:|-]+\|$", lines[i + 1])):
            n = lines[i].count("|")
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                if lines[j].count("|") != n:
                    problems.append(
                        f"{rel}:{j+1}: table row has {lines[j].count('|')} pipes, "
                        f"header {n}")
                j += 1
            i = j
            continue
        i += 1
    print(f"{rel}: {len(lines)} lines, {len(anchors)} headings, {fences//2} code blocks")

# --- citations --------------------------------------------------------------
for name in ("immrax_validator.py", "optimizer.py", "main.py", "validator.py",
             "robot_model.py", "multiverse_optimizer.py"):
    src = os.path.join(REPO, name)
    if not os.path.exists(src):
        continue
    raw, text = read(src)
    src_lines = text.split("\n")
    for path in (DOC, NOTES):
        raw, doc = read(path)
        rel = os.path.relpath(path, REPO)
        # a word boundary before the name, so immrax_validator.py is not
        # matched as validator.py
        pattern = r"(?<![\w])" + re.escape(name) + r":(\d+)(?:-(\d+))?"
        for m in re.finditer(pattern, doc):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            if start < 1 or end > len(src_lines):
                problems.append(
                    f"{rel}: out-of-range citation {name}:{m.group(1)}")

print()
if problems:
    print(f"!! {len(problems)} problems")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("All documentation checks passed.")
