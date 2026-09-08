"""Run the same five analytic problems through mincon.fmincon; write parity_mincon.json."""
import json, os, sys, time
import numpy as np
import mincon

HERE = os.path.dirname(os.path.abspath(__file__))

def problems():
    return [
        dict(name="P1_linear_ineq", fun=lambda x: float(((x - 1) ** 2).sum()), x0=[0, 0], A=[[1, 1]], b=[1]),
        dict(name="P2_nonlinear_ineq", fun=lambda x: float(x[0] + x[1]), x0=[0.5, 0.5],
             nonlcon=lambda x: ([x[0] ** 2 + x[1] ** 2 - 1], [])),
        dict(name="P3_nonlinear_eq", fun=lambda x: float(x[0] ** 2 + x[1] ** 2), x0=[2, 0.5],
             nonlcon=lambda x: ([], [x[0] * x[1] - 1])),
        dict(name="P4_bounds", fun=lambda x: float((x[0] - 3) ** 2 + (x[1] + 1) ** 2), x0=[1, 1], lb=[0, 0], ub=[2, 2]),
        dict(name="P5_ranged_linear", fun=lambda x: float((x[0] - 3) ** 2 + (x[1] - 3) ** 2), x0=[0, 0],
             A=[[1, 1], [-1, -1]], b=[2, -1]),
    ]

def main():
    runs = []
    for p in problems():
        for method in ("auto", "interior-point"):
            kw = {k: p[k] for k in ("A", "b", "Aeq", "beq", "lb", "ub", "nonlcon") if k in p}
            t0 = time.perf_counter()
            r = mincon.fmincon(p["fun"], p["x0"], options={"threads": 1} , **kw) if method == "auto" else \
                mincon.minimize(p["fun"], p["x0"], method=method, options={"threads": 1},
                                bounds=list(zip(p.get("lb", [None]*2), p.get("ub", [None]*2))) if ("lb" in p or "ub" in p) else None,
                                constraints=_scipy_cons(p))
            t = time.perf_counter() - t0
            runs.append(dict(problem=p["name"], algorithm=f"mincon-{method}", x=[float(v) for v in r.x], fun=float(r.fun),
                             status=int(r.status), success=bool(r.success), message=r.message, nit=int(r.nit),
                             nfev=int(r.nfev), maxcv=float(r.maxcv), optimality=float(r.optimality),
                             wall_seconds=t, solver_time=float(r.time), notes=list(r.notes)))
    out = dict(mincon_version=mincon.__version__, python=sys.version, runs=runs)
    with open(os.path.join(HERE, "parity_mincon.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote parity_mincon.json")

def _scipy_cons(p):
    cons = []
    if "A" in p:
        A, b = np.asarray(p["A"], float), np.asarray(p["b"], float)
        cons.append({"type": "ineq", "fun": lambda x, A=A, b=b: b - A @ x})
    if "nonlcon" in p:
        nl = p["nonlcon"]
        cons.append({"type": "ineq", "fun": lambda x, nl=nl: -np.asarray(nl(x)[0], float)})
        cons.append({"type": "eq", "fun": lambda x, nl=nl: np.asarray(nl(x)[1], float)})
    cons = [c for c in cons if np.asarray(c["fun"](np.asarray(p["x0"], float))).size > 0]
    return cons or None

if __name__ == "__main__":
    main()
