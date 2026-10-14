#!/usr/bin/env python3
"""Remove MathWorks runtime text and local identifiers from benchmark output.

The `fmincon` workers record what MATLAB prints: the full exit message of every
run (`native_message` in harness records, `message` in friction records), the
last warning, and, in console logs and the friction summaries, the warnings,
their stack traces (which name toolbox functions that are not part of MathWorks'
public interface) and each run's exit headline. None of that text
is needed to score a run (the oracle and every table use `native_status` /
`exitflag`, the point, the multipliers and the counts), and it is MathWorks'
text, so it is not published with the records (docs/IP_AUDIT.md, F-05 and
F-06). The same pass replaces the benchmark machine's host name and the user
directory with the placeholders `<host>` and `<home>` already used in the
experiment files.

The script only edits the text it targets, in place, so every number in a
record keeps its exact bytes. It is idempotent; `--check` changes nothing and
exits 1 if a file would change (run it before committing new results).

    python scripts/scrub_matlab_records.py bench/results               # on the benchmark machine
    python scripts/scrub_matlab_records.py bench/results --host NAME --user NAME
    python scripts/scrub_matlab_records.py bench/results --check

On Windows the host and user default to %COMPUTERNAME% and %USERNAME%, so the
first form is enough on the machine that produced the runs. The names are taken
from the command line or the environment and are never written into this file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

TEXT_SUFFIXES = {".jsonl", ".json", ".md", ".txt", ".log", ".csv"}
JSON_STRING = r'"(?:[^"\\]|\\.)*"'
# Fields of an fmincon record that hold MATLAB's own text.
MESSAGE_FIELDS = ("native_message", "message")
WARNINGS_FIELD = re.compile(r'("warnings"\s*:\s*)(?:\[\s*(?:' + JSON_STRING + r'\s*,?\s*)*\]|' + JSON_STRING + ')')
# A MATLAB warning's stack trace in a console log: "[\b> In fn ... In fn (line N)]\b", over several lines.
STACK_TRACE = re.compile(r"\[\x08?> In [^\]]*\]\x08?")
STACK_PLACEHOLDER = "[stack trace removed]"
# The warning itself, printed the same way: "[\bWarning: ...]\b".
WARNING = re.compile(r"\[\x08?Warning: [^\]]*\]\x08?")
WARNING_PLACEHOLDER = "[MATLAB warning removed]"
# A worker's own progress line for an fmincon run ends with MATLAB's exit headline after the wall time:
# "<problem>  fmincon-sqp  ok  exitflag=1 f_model=63 wall=0.166s <headline>". Keep the line, drop the headline.
RUN_LINE = re.compile(r"^(\S+[ \t]+fmincon-\S+[ \t]+\S+[ \t]+(?:exitflag|status)=\S+.*?wall=[0-9.]+s)[ \t]+[^\s][^\r\n]*", re.M)


def is_fmincon_record(line: str) -> bool:
    try:
        rec = json.loads(line)
    except ValueError:
        return False
    return isinstance(rec, dict) and str(rec.get("solver", "")).startswith("fmincon")


def scrub_record_line(line: str) -> str:
    for field in MESSAGE_FIELDS:
        line = re.sub(r'("' + field + r'"\s*:\s*)' + JSON_STRING, r"\1null", line)
    return WARNINGS_FIELD.sub(r"\1[]", line)


def scrub_summary_tables(text: str) -> str:
    """In a Markdown summary, the "message" column of every table under a "### fmincon..." heading."""
    out, section, message_col = [], "", None
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            section, message_col = stripped.lstrip("#").strip(), None
        elif stripped.startswith("|") and section.startswith("fmincon"):
            cells = stripped.strip("|").split("|")
            names = [c.strip() for c in cells]
            if "message" in names:
                message_col = names.index("message")
            elif message_col is not None and not set(stripped) <= set("|-: ") and message_col < len(cells):
                end = "\r" if line.endswith("\r") else ""
                cells[message_col] = " [removed] "
                line = "|" + "|".join(cells) + "|" + end
        elif not stripped:
            message_col = None
        out.append(line)
    return "\n".join(out)


def identifier_patterns(host: str | None, user: str | None) -> list[tuple[re.Pattern, str]]:
    pats = []
    if user:
        u = re.escape(user)
        # C:\Users\name as written in text, as escaped inside JSON, and with forward slashes.
        pats.append((re.compile(r"[A-Za-z]:(?:\\\\|\\|/)Users(?:\\\\|\\|/)" + u + r"(?![A-Za-z0-9_.-])", re.I), "<home>"))
    if host:
        pats.append((re.compile(r"(?<![A-Za-z0-9-])" + re.escape(host) + r"(?![A-Za-z0-9-])", re.I), "<host>"))
    return pats


def scrub_text(path: Path, text: str, pats) -> str:
    if path.suffix == ".jsonl":
        lines = text.split("\n")
        text = "\n".join(scrub_record_line(ln) if is_fmincon_record(ln) else ln for ln in lines)
    if path.suffix == ".log":
        text = STACK_TRACE.sub(STACK_PLACEHOLDER, text)
        text = WARNING.sub(WARNING_PLACEHOLDER, text)
        text = RUN_LINE.sub(r"\1", text)
    if path.suffix == ".md":
        text = scrub_summary_tables(text)
    for pat, repl in pats:
        text = pat.sub(repl, text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--host", default=os.environ.get("COMPUTERNAME"), help="host name to replace with <host>")
    ap.add_argument("--user", default=os.environ.get("USERNAME") if os.name == "nt" else None,
                    help="user directory name to replace with <home>")
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 if any file would change")
    args = ap.parse_args()
    pats = identifier_patterns(args.host, args.user)
    files = []
    for root in args.paths:
        if not root.exists():
            print(f"skipping {root}: no such file or directory")
            continue
        files.extend([root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file()))
    changed = 0
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            text = raw.decode("cp1252")
            encoding = "cp1252"
        new = scrub_text(path, text, pats)
        if new != text:
            changed += 1
            print(("would change " if args.check else "scrubbed ") + str(path))
            if not args.check:
                path.write_bytes(new.encode(encoding))
    print(f"{changed} file(s) {'would change' if args.check else 'changed'}")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
