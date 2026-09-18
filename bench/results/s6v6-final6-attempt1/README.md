# Round 6, first attempt: aborted, kept, not the round

September 18, 2026. The sealed run of `../s6v6-final6/SEAL.md` was launched at
14:50 and stopped by hand at 16:51 with 53 records of 414 written (track A only:
`mincon` 23, `mincon-ip` 23, `mincon-sqp` 7). **Nothing here is a result of
round 6.** The files are the run's own, moved unchanged out of the sealed
directory so that the run can start again from an empty one (`SHA256SUMS`);
the candidate, the set, the targets and the protocol are untouched, and no
solver was changed between the two launches.

## What happened to the instrument

From the Windows power log (Kernel-Power events 105, 506, 507):

| time | event |
|---|---|
| 14:50 | launch, on mains power, nobody on the machine |
| 14:57:44 | power button, then lid closed: Modern Standby |
| 14:57:54 | charger unplugged |
| 15:18:52 | a momentary wake, then standby again ("battery drain budget exceeded") |
| 16:25:21 | lid opened; from here on battery, Balanced plan, with an interactive session working on the same machine |
| 16:49:11 | charger plugged in again, during the `mincon-sqp` arm |
| 16:51 | run stopped by hand (last record written 16:50:35) |

Two consequences, either of which is enough to discard the attempt.

**A record that is the supervisor's, not the solver's.** `mincon` on
WINKLER_SOFT_250 is recorded as `timeout`, "killed by supervisor after 1298s
(hard limit 210s)". The supervisor polls once a second, so it cannot observe
1298 s against a limit of 210 s unless the clock jumps between two polls. It
did: the worker was frozen by the standby of 14:57:44 about half a minute
into the problem, and at the wake of 15:18:52 the supervisor saw 1298 s on
the clock and killed it. The solver was never given its 60 s.

**A machine whose speed moved by more than three times between arms.** The
cost of one model call (`time.callback / (f_model + c_model)`) does not depend
on the solver. Milliseconds per call, with the build time of the same model
and the objective evaluations that fitted into the 60 s:

| problem | sealed, quiet | mincon | mincon-ip | mincon-sqp |
|---|---:|---|---|---|
| EXPSPEC_90 | 0.76 | 2.77 (build 21 s, 20 966) | 2.70 (21 s, 21 839) | 1.17 (9 s, 49 983) |
| LOGGAS_120 | 1.80 | 1.58 (17 s, 19 115) | 2.22 (21 s, 13 724) | 1.54 (22 s, 19 125) |
| LONGMEM_S08_200 | | 4.25 (33 s, 13 633) | 14.26 (113 s, 3 819) | not run |
| LONGMEM_S20_200 | 4.01 | 4.52 (31 s, 12 687) | 12.44 (98 s, 4 469) | not run |
| TVDENOISE_1000 | 0.61 | 0.75 (13 s, 62 349) | 2.45 (38 s, 19 062) | 0.56 (9 s, 16 040) |
| TVDENOISE_500 | | 1.10 (23 s, 47 084) | 0.46 (7 s, 100 292) | not run |
| WINKLER_SOFT_250 | 3.89 | killed, see above | 5.54 (67 s, 10 793) | not run |
| WINKLER_STIFF_250 | 1.82 | 2.07 (15 s, 8 144) | 2.98 (19 s, 8 813) | not run |

The sealed column is the objective alone, timed in one quiet process
(`SEAL.md`); the others include the harness's counting wrapper and the
constraint calls, so compare a row across arms before comparing it with the
first column. The `mincon` arm's first sixteen problems ran before the
standby, on mains power, and sit near the sealed costs; its last six ran
after the wake on battery (TVDENOISE_500 at 1.10 ms against 0.46 ms in the
next arm, minutes later). The `mincon-ip` arm ran wholly on battery beside an
interactive session, and one of that session's commands, a recursive search
from the repository root that ran for about three minutes before it was
killed, overlapped LONGMEM_S08_200. The seven `mincon-sqp` records straddle
the return of the charger and are nearer the sealed costs. On a clock-bound problem
the number of evaluations that fit into 60 s decides attainment, and here
that number moved by 2.4 to 3.6 times with the power source. Rule 3 of
`docs/23`: a number produced by the measuring instrument rather than the
solver is not evidence about the solver.

## What these records were used for

Once, before the release of 0.2.0, which shipped before round 6 was scored
(`docs/23`, Phase F, says why): the 53 records were scored with targets v7 to
look for **false certificates only**, a claim of success at a point the oracle
rejects. A slow machine takes evaluations away; it cannot make a solver
certify a wrong point, so the records serve that purpose and no other.

The rule applied is the fixture gate's (`Expect::NoFalseCertificate`): success
claimed away from the target is a lie unless an independent stationarity check
passes. **None was found.** Of the 53 records 23 claim success
(`screen-scored-A.jsonl`, `screen-summary.txt`); all 23 points are feasible;
18 attain the sealed target; the other five are PHASESPLIT_6 (all three arms)
and PHASESPLIT_10 (`mincon`, `mincon-ip`), certified at f = 2.044469 and
f = 2.515220. Those are the values the generator recorded before sealing for
the minimum of the start's own basin, and the oracle confirms each point
first-order KKT with supplied and with recovered multipliers (stationarity
6e-8 to 3e-7, no dual-sign violation). `heldout6.py` said before any run
that a local solver which certifies that minimum "is correct and not
attained". `screen-summary.txt` prints them under the heading FALSE
CERTIFICATES because its first rule was cruder than the project's (claimed and
not attained); the file is kept as it was produced.

One success carries no certificate from the oracle: `mincon-ip` on
WING_SIMPLE, attained and feasible, with a recovered stationarity of 5.7e-4
and a dual-sign violation of 5.3e-2 on a problem whose rows run to 4e6
(`SEAL.md` names that scale as a weak spot of the set). It is an attained
point, so it is not a false certificate; it is listed because the screen saw
it. No budget exit and no `Acceptable` exit claimed success.

Nothing else may be read from these records: not attainment, not evaluation
counts, not wall time.
