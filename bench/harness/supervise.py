"""Experiment supervisor: runs solver workers with hard per-problem wall-time limits, resumably.

    python supervise.py --experiment s2-dev --solvers mincon,fmincon-interior-point --track A \
        --split dev --maxtime 60 --hard-factor 3 --out results/s2-dev

A worker (Python or MATLAB) processes a list of problems and reports the problem it is on through a
progress file. If a problem exceeds maxtime * hard_factor + grace, the worker process tree is killed,
a schema-complete `timeout` record is written for that problem (no invented point), and a fresh worker
continues with the remaining problems. Records already present in the output file are skipped, so a
multi-hour run is resumable. Problem order is a fixed-seed shuffle per (solver, repeat), reversed on
odd repeats, so run order is balanced across solvers.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import schema  # noqa: E402

MATLAB_SOLVERS = ("fmincon-interior-point", "fmincon-sqp", "fmincon-active-set")
DEFAULT_MATLAB = r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"


def select_problems(manifest_path, split=None, families=None, names=None, exclude_tags=("diagnostic",), max_n=None):
    man = json.load(open(manifest_path))["problems"]
    out = []
    for p in man:
        if names and p["name"] not in names:
            continue
        if split and p.get("split") != split and not names:
            continue
        if families and p["family"] not in families:
            continue
        if exclude_tags and any(t in p.get("tags", []) for t in exclude_tags):
            continue
        if max_n and p["n"] > max_n:
            continue
        out.append(p["name"])
    return out


def existing_keys(path):
    keys = set()
    if os.path.exists(path):
        for r in schema.read_jsonl(path):
            keys.add((r["problem"], r["solver"], r["track"], int(r.get("repeat", 0))))
    return keys


def kill_tree(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def launch(solver, track, problems, out_file, progress_file, a, repeat):
    plist = ",".join(problems)
    if solver in MATLAB_SOLVERS:
        cmd = [a.matlab, "-batch",
               f"worker_matlab('{solver}','{track}','{plist}','{out_file}','{progress_file}',{a.maxtime},{a.maxfev},'{a.experiment}',{repeat},{a.threads})"]
        cwd = HERE
    else:
        cmd = [a.python, os.path.join(HERE, "worker_python.py"), "--solver", solver, "--track", track, "--problems", plist,
               "--out", out_file, "--progress", progress_file, "--maxtime", str(a.maxtime), "--maxfev", str(a.maxfev),
               "--threads", str(a.threads), "--experiment", a.experiment, "--repeat", str(repeat)]
        cwd = HERE
    log = open(out_file + ".log", "a")
    kw = {}
    if os.name != "nt":
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, **kw)


def read_progress(path):
    try:
        with open(path) as fh:
            parts = fh.read().split()
        return parts[0], float(parts[1])
    except Exception:  # noqa: BLE001
        return None, None


def timeout_record(problem, solver, track, a, repeat, seconds, man):
    p = man[problem]
    return schema.new_record(experiment=a.experiment, track=track, problem=problem, family=p["family"], split=p.get("split"),
                             checksum=p["checksum"], n=p["n"], m=p["m"], solver=solver,
                             derivatives=("exact" if track == "C" else "fd"), threads=a.threads,
                             budget=dict(maxtime=a.maxtime, maxfev=a.maxfev), outcome="timeout",
                             error=f"killed by supervisor after {seconds:.0f}s (hard limit {a.maxtime * a.hard_factor + a.grace:.0f}s)",
                             time=dict(solve_wall=seconds), repeat=repeat)


def run_solver(solver, track, problems, a, man, repeat):
    tag = solver.replace("@", "_").replace(",", "_").replace("=", "-")
    out_file = os.path.join(a.out, f"{tag}.{track}.jsonl")
    progress_file = os.path.join(a.out, f"{tag}.{track}.progress")
    done = existing_keys(out_file)
    pending = [p for p in problems if (p, solver, track, repeat) not in done]
    rng = random.Random(f"{a.seed}-{solver}-{repeat}")
    rng.shuffle(pending)
    if repeat % 2 == 1:
        pending.reverse()
    hard = a.maxtime * a.hard_factor + a.grace
    print(f"[{solver}/{track}] {len(pending)} pending of {len(problems)} (hard limit {hard:.0f}s each)", flush=True)
    while pending:
        with open(progress_file, "w") as pf:
            pf.write("")
        proc = launch(solver, track, pending, out_file, progress_file, a, repeat)
        launched_at = time.time()
        killed = None
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            cur, t0 = read_progress(progress_file)
            now = time.time()
            if cur in pending:
                # t0 is the worker's clock (MATLAB writes datenum seconds, not epoch); trust it only when sane
                started = t0 if (t0 is not None and abs(now - t0) < 3600) else launched_at
                started = max(started, launched_at)
                if now - started > hard:
                    kill_tree(proc)
                    killed = (cur, now - started)
                    break
            time.sleep(1.0)
        finished = set(k[0] for k in existing_keys(out_file) if k[1] == solver and k[2] == track and k[3] == repeat)
        if killed:
            name, secs = killed
            with open(out_file, "a") as fh:
                fh.write(schema.dumps(timeout_record(name, solver, track, a, repeat, secs, man)) + "\n")
            finished.add(name)
            print(f"[{solver}/{track}] TIMEOUT {name} after {secs:.0f}s", flush=True)
        remaining = [p for p in pending if p not in finished]
        if not killed and remaining and proc.returncode not in (None, 0):
            # worker crashed before finishing: record a crash for the problem it was on, continue with the rest
            cur, _ = read_progress(progress_file)
            if cur in remaining:
                with open(out_file, "a") as fh:
                    r = timeout_record(cur, solver, track, a, repeat, 0.0, man)
                    r["outcome"] = "crash"; r["error"] = f"worker exited with code {proc.returncode} while on {cur}"
                    fh.write(schema.dumps(r) + "\n")
                remaining.remove(cur)
                print(f"[{solver}/{track}] CRASH {cur} (rc={proc.returncode})", flush=True)
            elif remaining and not killed:
                print(f"[{solver}/{track}] worker rc={proc.returncode} with {len(remaining)} remaining; giving up on this solver", flush=True)
                break
        if not killed and remaining and proc.returncode == 0:
            print(f"[{solver}/{track}] worker exited 0 but {len(remaining)} problems missing; giving up", flush=True)
            break
        pending = remaining
    return out_file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--solvers", required=True)
    ap.add_argument("--track", default="A")
    ap.add_argument("--split", default=None)
    ap.add_argument("--families", default=None)
    ap.add_argument("--problems", default=None)
    ap.add_argument("--include-diagnostic", action="store_true")
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--maxtime", type=float, default=60.0)
    ap.add_argument("--maxfev", type=int, default=100000)
    ap.add_argument("--hard-factor", type=float, default=3.0)
    ap.add_argument("--grace", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", default="s1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--matlab", default=DEFAULT_MATLAB)
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(HERE), "corpus", "manifest.json"))
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    a.manifest = os.path.abspath(a.manifest)
    os.makedirs(a.out, exist_ok=True)
    man = {p["name"]: p for p in json.load(open(a.manifest))["problems"]}
    problems = select_problems(a.manifest, split=a.split, families=a.families.split(",") if a.families else None,
                               names=set(a.problems.split(",")) if a.problems else None,
                               exclude_tags=() if a.include_diagnostic else ("diagnostic",), max_n=a.max_n)
    with open(os.path.join(a.out, "experiment.json"), "a") as fh:
        # Record the arguments without the machine's home directory (published logs must not carry private paths).
        home = os.path.expanduser("~")
        args = {k: (v.replace(home, "<home>") if isinstance(v, str) else v) for k, v in vars(a).items()}
        fh.write(json.dumps(dict(args=args, problems=problems, started=time.time())) + "\n")
    for repeat in range(a.repeats):
        for solver in a.solvers.split(","):
            run_solver(solver, a.track, problems, a, man, repeat)
    print("done", flush=True)


if __name__ == "__main__":
    main()
