# Friction audit at the Phase C wheel: nothing changes without the option (September 18, 2026)

The audit of `../s7-friction-b` (fifteen realistic problems, a user's minimal inputs, five solvers,
every returned point judged by the independent oracle) re-run at the wheel of the Phase C tree
(`docs/22_ROUND5_ROBUSTNESS_PLAN.md` section 7.17; wheel sha256 `2f878929e60ce53e`, whose extension
module and `__init__.py` are byte for byte those of the wheel of record, `444c3ac506ce0ccc`, built
after the packaged README was updated). Phase C adds
parallel and batched finite-difference probes behind `workers=` and `vectorized=`. The audit uses
neither, and its question is whether a solve that uses neither is the solve it was before.

**It is.** mincon attains 14/15 as before, and all fifteen mincon records are identical to
`s7-friction-b` in every field that is not a clock (`started`, `callback_seconds`, `model_time`,
`solver_time`): the returned point to the last digit, the objective, the evaluation and call
counts, the iterations, the status, the message, the notes and the verdict. One field differs in
text: `wrong_gradient` is the refusal, its record stores the traceback, and the traceback quotes
lines of mincon's own `__init__.py`, which moved; the exception it ends in is identical. The four
other solvers' sixty records are identical in every non-clock field as well, so the machine, SciPy
and MATLAB did not move either. MATLAB and Python definitions agree at x0 to 3.6e-11 relative over
all 30 MATLAB records.

| solver | attained |
|---|---|
| mincon (`fmincon` facade, defaults) | 14/15 |
| fmincon-interior-point | 12/15 |
| fmincon-sqp | 12/15 |
| SciPy SLSQP | 11/15 |
| SciPy trust-constr | 12/15 |

`summary.md` and `scored.jsonl` are the scored output, `*.jsonl` the raw records, `matlab_run.log`
the MATLAB run. What the options do when they are on is measured in `docs/22` section 7.17, not
here.
