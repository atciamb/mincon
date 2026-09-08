# Benchmark harness v2

Pipeline (all paths relative to the repository root):

1. `python bench/corpus/generate.py` — regenerate `bench/corpus/matlab/*.m`,
   `manifest.json` (with the frozen split) and `probes.json` from the SymPy
   definitions in `hs.py`, `adversarial.py`, `structured.py`.
2. `matlab -batch "run('bench/corpus/equivalence_matlab.m')"` then
   `python bench/corpus/equivalence.py` — cross-language equivalence gate.
3. `python bench/harness/targets.py init` — `targets_v1.json` (closed-form and
   published targets; best-known targets are set by `targets.py update`).
4. `python bench/harness/supervise.py --experiment NAME --solvers mincon,fmincon-interior-point,fmincon-sqp,scipy-slsqp --track A --split dev --out bench/results/NAME --maxtime 60 --python <venv python>`
   — runs the workers (`worker_python.py`, `worker_matlab.m`) with hard
   per-problem limits; resumable; `--repeats k` for timing replicates;
   `mincon@key=value,...` for option ablations.
5. `python bench/harness/score.py bench/results/NAME/*.A.jsonl -o bench/results/NAME/scored-A.jsonl`
   — attaches the independent oracle verdict (`oracle.py`) under the frozen
   target version.
6. `python bench/harness/analyze.py bench/results/NAME/scored-A.jsonl --baseline fmincon-interior-point`
   — attainment, paired win/loss, family-bootstrap intervals, cost ratios,
   performance-profile points; `timing.py` for wall-time with replicates.

Tests: `python -m pytest bench/harness/test_oracle.py`.
Requirements: numpy, scipy, sympy, pytest; MATLAB R2025b + Optimization
Toolbox for the fmincon workers (not needed for mincon-only runs).
