"""Independent verification of MATLAB and mincon parity runs against closed-form solutions."""
import json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
S = 1 / math.sqrt(2)
TRUTH = {  # closed-form: (x*, f*), plus an independent feasibility function (max violation, <=0 form incl. bounds)
    "P1_linear_ineq": ([0.5, 0.5], 0.5, lambda x: x[0] + x[1] - 1),
    "P2_nonlinear_ineq": ([-S, -S], -math.sqrt(2), lambda x: x[0]**2 + x[1]**2 - 1),
    "P3_nonlinear_eq": ([[1.0, 1.0], [-1.0, -1.0]], 2.0, lambda x: abs(x[0]*x[1] - 1)),
    "P4_bounds": ([2.0, 0.0], 2.0, lambda x: max(-x[0], -x[1], x[0]-2, x[1]-2)),
    "P5_ranged_linear": ([1.0, 1.0], 8.0, lambda x: max(x[0]+x[1]-2, 1-x[0]-x[1])),
}
def main():
    rows = []
    for fn in ("parity_matlab.json", "parity_mincon.json"):
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            print("missing", fn); continue
        d = json.load(open(path))
        for r in d["runs"]:
            xs, fs, feas = TRUTH[r["problem"]]
            x = np.asarray(r["x"], float)
            cands = xs if isinstance(xs[0], list) else [xs]   # several global minimizers are legitimate
            xerr = min(float(np.max(np.abs(x - np.asarray(c)))) for c in cands)
            ferr = abs(r["fun"] - fs)
            viol = max(0.0, float(feas(x)))
            ok = xerr < 1e-4 and ferr < 1e-5 and viol <= 1e-6
            flag = r.get("exitflag", r.get("status"))
            rows.append((r["problem"], r["algorithm"], flag, xerr, ferr, viol, r.get("funcCount", r.get("nfev")), r["wall_seconds"], ok))
    print(f"{'problem':<18} {'solver':<22} {'flag':>4} {'|x-x*|':>9} {'|f-f*|':>9} {'viol':>9} {'nfev':>5} {'wall_s':>8}  verdict")
    for row in rows:
        print(f"{row[0]:<18} {row[1]:<22} {row[2]:>4} {row[3]:9.2e} {row[4]:9.2e} {row[5]:9.2e} {row[6]:>5} {row[7]:8.4f}  {'PASS' if row[8] else 'FAIL'}")
    print(f"{sum(r[8] for r in rows)}/{len(rows)} runs reached the closed-form solution (|x-x*|<1e-4, |f-f*|<1e-5, viol<=1e-6; thresholds fixed before the second run, see README)")
if __name__ == "__main__":
    main()
