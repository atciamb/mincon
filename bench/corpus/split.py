"""Frozen family-aware split assignment (development / validation / final / diagnostic).

Rules, fixed before any matched run (see docs/15_BENCHMARK_PROTOCOL_V2.md):
* the 44 Hock–Schittkowski problems already present in the Rust test set are exposed -> development;
* the remaining HS problems are hashed by name into dev/val/final at 40/30/30;
* adversarial problems are hashed at 50/25/25; diagnostic-tagged problems go to the diagnostic track;
* whole structured families are assigned together: chainrosen, lqtraj, expfit -> dev;
  quadsphere -> validation; ellipsoid, portfolio -> final;
* engineering designs: SPRING, THREEBAR_TRUSS, CANTILEVER -> validation; PRESSURE_VESSEL, WELDED_BEAM,
  SPEED_REDUCER -> final.
The assignment is a pure function of the problem name; it never looks at any solver result.
"""
import hashlib

EXPOSED_HS = {"HS1", "HS2", "HS3", "HS4", "HS5", "HS38", "HS45", "HS110", "HS6", "HS7", "HS8", "HS9", "HS26", "HS27", "HS28",
              "HS39", "HS40", "HS42", "HS10", "HS11", "HS12", "HS13", "HS14", "HS15", "HS16", "HS18", "HS21", "HS22", "HS23",
              "HS24", "HS29", "HS30", "HS31", "HS32", "HS33", "HS34", "HS35", "HS36", "HS37", "HS41", "HS43", "HS44", "HS71", "HS100"}
# Round 1 (2026-09-07): dev / validation / final. The round-1 final split was used for diagnosis
# after its qualification run (protocol §2), so it is relabelled "final1-dev" here: development
# material that is reported separately. Round 2 held-out families (heldout2.py) are "final2".
FAMILY_SPLIT = {"chainrosen": "dev", "lqtraj": "dev", "expfit": "dev", "quadsphere": "validation", "ellipsoid": "final1-dev",
                "portfolio": "final1-dev", "polyqp": "final2", "dispatch": "final2", "catenary": "final2", "ellipsoid2": "final2",
                "quadsphere2": "final2", "expfit2": "final2", "engineering2": "final2",
                # Round 3 held-out families (heldout3.py), generated after candidate C6 was frozen.
                "nnls_simplex": "final3", "maxent": "final3", "logsumexp": "final3", "rosen_sphere": "final3",
                "sinfit": "final3",
               # Round 4 held-out families (heldout4.py), generated after candidate C7 was frozen.
               "covqp": "final4", "denselap": "final4", "snl": "final4", "obstacle": "final4",
               # Round 5 held-out families (heldout5.py), generated after the round-5 increments were frozen.
               "pkfit": "final5", "logistic": "final5", "tcport": "final5", "deconv": "final5", "noisyqp": "final5",
               "ncboxqp": "final5", "engineering3": "final5",
               # Round 6 held-out families (heldout6.py), generated after the tree was frozen for round 6.
               "expspec": "final6", "loggas": "final6", "stiefel": "final6", "cstr": "final6", "propfair": "final6",
               "tvdenoise": "final6", "gpdesign": "final6", "stepnoise": "final6", "phasesplit": "final6",
               "winkler": "final6", "longmem": "final6"}
ENGINEERING = {"SPRING": "validation", "THREEBAR_TRUSS": "validation", "CANTILEVER": "validation",
               "PRESSURE_VESSEL": "final1-dev", "WELDED_BEAM": "final1-dev", "SPEED_REDUCER": "final1-dev"}
NEW_HS = {"HS25", "HS56", "HS84"}   # added after round 1: never seen by tuning
ROUND3_HS = {"HS114"}               # added after candidate C6 was frozen


def _u(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) / 2 ** 32


def assign(name, family, tags):
    if "diagnostic" in tags:
        return "diagnostic"
    if family == "hs":
        if name in ROUND3_HS:
            return "final3"
        if name in NEW_HS:
            return "final2"
        if name in EXPOSED_HS:
            return "dev"
        u = _u(name)
        return "dev" if u < 0.4 else ("validation" if u < 0.7 else "final1-dev")
    if family == "adversarial":
        u = _u(name)
        return "dev" if u < 0.5 else ("validation" if u < 0.75 else "final1-dev")
    if family == "engineering":
        return ENGINEERING[name]
    return FAMILY_SPLIT[family]
