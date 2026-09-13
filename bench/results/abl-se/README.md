# abl-se: counted pivot signs in the interior-point KKT factorisation (S-E), whole-corpus ablation, September 12, 2026

Study S-E of `docs/22` (§7.5, §7.8): with an exact Lagrangian Hessian the interior-point member
took 56 iterations on HS71 where quasi-Newton takes 10, because the sparse `LDL^T` expects every
primal pivot to be positive and perturbs a legitimate negative pivot of an indefinite primal block
to `+1e-7`, which voids the inertia certificate and forces `delta_w` at every iteration although the
reduced Hessian is positive and the true inertia is already (n, m). `Options::kkt_pivot_signs`
(`Auto` / `Expected` / `Free`, Python `kkt_pivot_signs`) lets the factorisation keep the pivots'
signs and count the inertia instead; `Auto`, the default, does that only when the member uses the
model's exact Hessian and keeps the expectations with a quasi-Newton one. HS71 with `hess=` under
`method='interior-point'`: 56 -> 10 iterations, certified; forcing `Expected` reproduces the 56.

Run: track A, all 169 non-diagnostic corpus problems, defaults otherwise, single thread,
60 s / 100 000 evaluations, targets v5, wheel `wheels-se` (the HEAD tree plus S-E). The corpus
supplies no Hessians, so under `Auto` the option cannot touch a record; the control arm checks
that, and the two `Free` arms force the counted signs onto the quasi-Newton path, which is the
only way the code can change a corpus record. Baseline: the current default's records (`abl-c7`
and `s6v4-final4` rule-off arms), copied into `cmp-*.jsonl` under their original solver names.

| arm | attained candidate / baseline (of 166 scorable) | cost ratio [95 % family bootstrap] | records that changed |
|---|---|---|---|
| portfolio, `mincon@kkt_pivot_signs=auto` (control) | 158 / 158 | 1.01 [1.00, 1.02] on 158 | three, all known: the two 60 s budget exits (COVQP_300, DENSELAP_250) and HS117 (the `scale_variables='auto'` default of `abl-i8`, attained either way) |
| portfolio, `mincon@kkt_pivot_signs=free` | 158 / 158 | 1.01 [1.00, 1.02] on 158 | the same three: the sign expectations never decided a record on the quasi-Newton path |
| IP member, `mincon-ip@kkt_pivot_signs=free` | 157 / 156 | 1.02 [1.00, 1.02] on 156 | seven: the two 60 s exits (COVQP_300 landed inside the target this time, a machine-speed effect), HS117, and HS16 / HS17 / UNITS (I1's changes, `abl-i1`); **RANKLOSS_JAC** is the one record the counted signs changed: a step-tolerance exit (status 6) became `Optimal` at 215 against 213 evaluations, attained either way |

Verdict: the falsifier of `docs/22` §7.5 ("HS71 at most 10 iterations, no corpus record change")
holds. Under the default `Auto` no record changes; even with the counted signs forced onto the
quasi-Newton path nothing is lost and one rank-deficient problem certifies where it stopped on
the step tolerance. `Auto` stays the default (the exact-Hessian path counts, the quasi-Newton path
keeps its expectations, which the measurement says are harmless rather than necessary).

Files as in `abl-i1` (`cmp-auto`, `cmp-free`, `cmp-ip-free`, `scored-*` and their analyses,
`diff-*.txt` from `diff_arms.py`, `score-*.log`, `run.log`).
