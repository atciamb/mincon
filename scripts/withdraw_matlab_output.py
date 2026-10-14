#!/usr/bin/env python3
"""Withdraw MATLAB-produced benchmark output from the tree (docs/IP_AUDIT.md, F-01).

This project no longer runs MATLAB. Everything a MATLAB session produced for it, and every number
computed from that output, leaves the current tree; the history keeps all of it unchanged, and
bench/results/MATLAB_WITHDRAWN.md (written by this script) lists each path, what was done to it
and the blob it had, so any original can be read with `git show <commit>:<path>`.

* Records: a JSONL file of fmincon records only is removed (`git rm`); in a file that mixes fmincon records
  with others, the fmincon lines are removed and every other line keeps its bytes. A record is
  an fmincon record when its "solver" starts with "fmincon" ("mincon-fmincon" is mincon's own
  facade and stays).
* MATLAB console and progress files (fmincon*.log, fmincon*.progress, matlab_run.log),
  bench/parity/parity_matlab.json, the wall-time comparisons against fmincon (timing.md,
  timing-vs-*.md) and the analyses whose baseline is an fmincon arm (*analysis-vs-fmincon*)
  are deleted.
* Reports that tabulate fmincon next to other solvers (the analyze.py reports, the friction
  summaries, the ablation tables): fmincon rows, columns and per-solver sections are removed.
  A paired comparison against an fmincon baseline, and performance-profile fractions computed
  with fmincon in the set, are replaced by a one-line note, because every number in them
  depends on fmincon. Everything else keeps its bytes.

Prose (READMEs, docs) is not edited here. The sealed files of bench/results/s6v6-final6/SEAL.md
are never touched. Idempotent: a second run changes nothing.

    python scripts/withdraw_matlab_output.py            # apply, and write the register
    python scripts/withdraw_matlab_output.py --dry-run  # list what would change
    python scripts/withdraw_matlab_output.py --preview DIR   # write the filtered files under DIR only
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTER_MD = ROOT / "bench/results/MATLAB_WITHDRAWN.md"
REGISTER_JSON = ROOT / "bench/results/matlab_withdrawn.json"
NOTE_PAIRED = "Withdrawn: this comparison was against an `fmincon` arm (`bench/results/MATLAB_WITHDRAWN.md`)."
NOTE_PROFILE = ("Withdrawn: these fractions were computed with `fmincon` arms in the comparison set "
                "(`bench/results/MATLAB_WITHDRAWN.md`).")
DELETE_NAMES = re.compile(r"^(fmincon[^/]*\.(log|progress)|matlab_run\.log|timing\.md|timing-vs-[^/]*\.md|"
                          r"[^/]*analysis-vs-fmincon[^/]*)$")
SKIP_NAMES = {"README.md", "SEAL.md", "PROTOCOL_V3A.md", "MATLAB_WITHDRAWN.md"}


def is_fm(name: str) -> bool:
    return name.strip().strip("`*").startswith("fmincon")


def tracked(*roots: str) -> list[Path]:
    out = subprocess.check_output(["git", "ls-files", "-z", *roots], cwd=ROOT).decode().split("\0")
    return [ROOT / p for p in out if p]


def blob(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", f"HEAD:{path.relative_to(ROOT).as_posix()}"],
                                   cwd=ROOT).decode().strip()


def read(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252"), "cp1252"


def filter_jsonl(text: str) -> tuple[str, int, int]:
    keep, removed, kept = [], 0, 0
    for line in text.splitlines(keepends=True):
        body = line.strip()
        if body:
            try:
                rec = json.loads(body)
            except ValueError:
                rec = None
            if isinstance(rec, dict) and is_fm(str(rec.get("solver", ""))):
                removed += 1
                continue
            kept += 1
        keep.append(line)
    return "".join(keep), removed, kept


def split_row(line: str) -> list[str]:
    return line.strip().strip("|").split("|")


def filter_report(text: str) -> str:
    """Remove fmincon from an analyze.py report, a friction summary or an ablation table."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    drop_cols: list[int] = []
    in_table = False
    while i < len(lines):
        line = lines[i]
        eol = line[len(line.rstrip("\r\n")):]
        s = line.strip()
        if s.startswith("#"):
            title = s.lstrip("#").strip()
            level = len(s) - len(s.lstrip("#"))
            # a section about one fmincon arm: drop it up to the next heading of the same or a higher level
            if is_fm(title):
                i += 1
                while i < len(lines):
                    t = lines[i].strip()
                    if t.startswith("#") and len(t) - len(t.lstrip("#")) <= level:
                        break
                    i += 1
                continue
            out.append(line)
            i += 1
            m = re.match(r"Paired comparison against `([^`]+)`", title)
            if (m and is_fm(m.group(1))) or title.startswith("Performance profile points"):
                # keep the heading; the body goes if it depends on fmincon
                j = i
                while j < len(lines) and not lines[j].strip().startswith("#"):
                    j += 1
                body = "".join(lines[i:j])
                if (m and is_fm(m.group(1))) or re.search(r"\|\s*fmincon", body):
                    out.append(eol or "\n")
                    out.append((NOTE_PAIRED if m else NOTE_PROFILE) + (eol or "\n"))
                    out.append(eol or "\n")
                    i = j
            in_table = False
            continue
        if s.startswith("|"):
            cells = split_row(s)
            if not in_table:
                in_table = True
                drop_cols = [k for k, c in enumerate(cells) if is_fm(c)]
            elif cells and is_fm(cells[0]):
                i += 1
                continue
            if drop_cols:
                cells = [c for k, c in enumerate(cells) if k not in drop_cols]
                line = "|" + "|".join(cells) + "|" + eol
            out.append(line)
            i += 1
            continue
        in_table = False
        m = re.match(r"^(\d+ problems, solvers: )(.*?)(; duplicates overwritten: \d+)\s*$", s)
        if m:
            names = [n for n in m.group(2).split(", ") if not is_fm(n)]
            line = m.group(1) + ", ".join(names) + m.group(3) + eol
        elif re.match(r"^- [^/]+ / fmincon", s):
            i += 1
            continue
        out.append(line)
        i += 1
    return "".join(out)


def filter_analysis_json(text: str) -> str:
    d = json.loads(text)
    if not isinstance(d, dict) or "solvers" not in d:
        return text
    solvers = list(d["solvers"])
    paired = d.get("paired", {})
    baseline = [s for s in solvers if s not in paired]
    d["solvers"] = {k: v for k, v in d["solvers"].items() if not is_fm(k)}
    if baseline and is_fm(baseline[0]):
        d["paired"] = {}
        d["paired_withdrawn"] = NOTE_PAIRED
    elif isinstance(paired, dict):
        d["paired"] = {k: v for k, v in paired.items() if not is_fm(k)}
    new = json.dumps(d, indent=1)
    return new + ("\n" if text.endswith("\n") else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preview", type=Path, default=None,
                    help="write the filtered files under this directory instead of in place; change nothing")
    args = ap.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT).decode().strip()
    entries, to_remove = [], []
    for path in tracked("bench/results", "bench/parity"):
        rel = path.relative_to(ROOT).as_posix()
        if path.name in SKIP_NAMES or rel.startswith("bench/results/s6v6-final6/"):
            continue
        if not path.exists():
            continue
        text, enc = read(path)
        action, detail, new = None, "", None
        if rel == "bench/parity/parity_matlab.json" or (rel.startswith("bench/results/") and DELETE_NAMES.match(path.name)
                                                      and "fmincon" in text.lower() + path.name):
            action = "deleted"
        elif path.suffix == ".jsonl":
            filtered, removed, kept = filter_jsonl(text)
            if removed and not kept:
                action, detail = "deleted", f"{removed} fmincon records"
            elif removed:
                action, detail, new = "filtered", f"{removed} fmincon records removed, {kept} kept", filtered
        elif path.suffix == ".json" and path.name.endswith(".analysis.json") and "fmincon" in text:
            new = filter_analysis_json(text)
            action, detail = "filtered", "fmincon solvers removed from the analysis"
        elif path.suffix in (".md", ".txt") and re.search(r"\|\s*`?fmincon|solvers: [^\n]*fmincon|### fmincon", text):
            new = filter_report(text)
            action, detail = "filtered", "fmincon rows, columns and sections removed"
        if action is None or (new is not None and new == text):
            continue
        entries.append(dict(path=rel, action=action, detail=detail, blob=blob(path)))
        print(f"{action:9} {rel}  {detail}")
        if args.dry_run:
            continue
        if args.preview:
            if new is not None:
                dest = args.preview / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(new.encode(enc))
            continue
        if action == "deleted":
            to_remove.append(rel)
        else:
            path.write_bytes(new.encode(enc))
    if to_remove and not args.dry_run and not args.preview:
        # every one of these is tracked and committed, so `git checkout <before> -- <path>` restores it
        subprocess.check_call(["git", "rm", "--quiet", "--", *to_remove], cwd=ROOT)
    print(f"{len(entries)} file(s) {'would change' if (args.dry_run or args.preview) else 'changed'}")
    if args.dry_run or args.preview or not entries:
        return 0
    old = json.loads(REGISTER_JSON.read_text()) if REGISTER_JSON.exists() else {"entries": []}
    seen = {e["path"] for e in entries}
    allentries = [e for e in old["entries"] if e["path"] not in seen] + [dict(e, before=head) for e in entries]
    REGISTER_JSON.write_text(json.dumps({"entries": sorted(allentries, key=lambda e: e["path"])}, indent=1) + "\n")
    write_register(sorted(allentries, key=lambda e: e["path"]))
    return 0


def write_register(entries: list[dict]) -> None:
    n_del = sum(e["action"] == "deleted" for e in entries)
    n_fil = len(entries) - n_del
    rows = "\n".join(f"| `{e['path']}` | {e['action']} | {e['detail']} | `{e['before']}` | `{e['blob'][:12]}` |"
                     for e in entries)
    REGISTER_MD.write_text(f"""# MATLAB output withdrawn from the tree

October 14, 2026. This project no longer runs MATLAB, and the output that MATLAB sessions
produced for it, with the numbers computed from that output, is no longer distributed in the
current tree: {n_del} files deleted and {n_fil} filtered, listed below. Nothing was rewritten in
the history, so each original is still readable there: `git show <before>:<path>` (the
commit named in the "before" column is the last one that has the file as it was), or the blob
named in the last column. The commit `cddf174` holds every one of them before any cleaning.

What stays, on purpose:

* every mincon and SciPy record and every number computed from them alone; where a report
  mixed them with fmincon, only the fmincon rows, columns and sections went;
* the experiment files (`experiment.json`), which record what was run, including the fmincon
  arms' names, and the sealed round-6 files of `s6v6-final6/SEAL.md`, which are not edited;
* `bench/corpus/matlab_probes.json` and `bench/harness/worker_matlab.m`, both sealed, until the
  round-6 report is committed;
* three development-split targets in `bench/corpus/targets_v3.json` to `targets_v7.json`
  (CATENARY_40, DISPATCH_20, EXPFIT2_NOISY) that were set from reference runs: v7 is sealed;
* descriptions of `fmincon` taken from MathWorks' public documentation, quoted briefly and
  attributed (`docs/20`).

Statements in the READMEs and docs that rested on the withdrawn numbers are withdrawn in place,
marked as withdrawn. `scripts/scrub_matlab_records.py --check` (CI job `records`) fails if
MATLAB output enters the tree again. Machine-readable copy: `matlab_withdrawn.json`. Written by
`scripts/withdraw_matlab_output.py`.

| path | action | what | before | original blob |
|---|---|---|---|---|
{rows}
""")


if __name__ == "__main__":
    sys.exit(main())
