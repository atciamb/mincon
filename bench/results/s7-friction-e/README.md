# Friction audit, mincon's records at the Phase D freeze (September 18, 2026)

The fifteen problems of `../s7-friction-d`, mincon only (`mincon-fmincon`, the facade at
defaults), at wheel `d57902fa79492531`, after `quadratic_bands='decaying'` became the default
(`docs/22_ROUND5_ROBUSTNESS_PLAN.md` section 7.19). The other four solvers did not change and were
not re-run; their records, the MATLAB twin's check and the scoring are those of `../s7-friction-d`.

Compared with `../s7-friction-d/mincon-fmincon.jsonl` field by field, the clock fields excluded
(`started`, `callback_seconds`, `model_time`, `solver_time`, `wall`): **all fifteen records are
identical**, 14 usable of 15, except one stored string: `wrong_gradient` is the refusal, its record
keeps the traceback, and the traceback quotes line numbers of mincon's own `__init__.py`, which
moved. No friction problem sets a budget the probe's dense build does not fit, so the new default
is never reached.
