# abl-phased45: the first form of `zero_step='decrease'`, falsified and stopped (September 18, 2026)

Phase D candidate 4 (`docs/22_ROUND5_ROBUSTNESS_PLAN.md` section 7.18). The first rule written
treated a feasible SQP step whose predicted merit decrease is below the rounding noise of the merit
function exactly like a vanished step: adopt the QP's multipliers, re-run the termination test, and
if it still fails classify the exit. It fixed the problem it was written for (`heatflux_design`:
484 evaluations to 104). This directory is what the corpus said (wheel `7f38e42de04da38a`, track A,
60 s / 100 000 evaluations, single thread; the default arm complete, the option arm stopped at 148
of 185 records, the two further arms never run; unscored, compared on raw status and counts with
`.local-research/phase_d/rawdiff.py`):

**Six problems lost their certificate.** HS100, HS110, HS25, LOGISTIC_NOISY2, PKFIT_NOISY2 and
THREEBAR_TRUSS went from `Optimal` to `Acceptable` ("The QP step vanished at a feasible point
(scaled KKT error 9.73e-6)"), a few iterations before the old path's certified exit. Fifteen other
records saved evaluations. Attainment would not have shown the loss; `success` does, and success
means certified, so the run was stopped and the rule replaced (`../abl-phased45b`): the condition
now only adopts the multipliers and re-tests, and is never a reason to stop.
