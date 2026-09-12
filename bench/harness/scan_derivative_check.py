"""Run every corpus problem with its exact derivatives through the start-up derivative check only
(a one-iteration solve): any RuntimeError mentioning "disagree" is a false alarm of the directional
check (docs/22 I3, H3 falsifier), any "Derivative check" note a mild one. Run from any directory with
the venv that has the candidate wheel installed; takes about 7 minutes.
"""
import sys, os, json, numpy as np, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "corpus"))
import spec, families
from model import CountingModel
import mincon
man = json.load(open(os.path.join(os.path.dirname(HERE), "corpus", "manifest.json")))["problems"]
gross, mild, ok, errs = [], [], 0, []
t0 = time.time()
for p in man:
    name = p["name"]
    s = spec.get(name); pn = s.numpy(); m = CountingModel(pn); g = m.row_groups()
    cons = []
    if pn.m:
        if g["eq"].size: i=g["eq"]; cons.append({"type":"eq","fun":lambda x,i=i: m.c(x)[i]-pn.cl[i],"jac":lambda x,i=i: m.jac(x)[i]})
        if g["lo"].size: i=g["lo"]; cons.append({"type":"ineq","fun":lambda x,i=i: m.c(x)[i]-pn.cl[i],"jac":lambda x,i=i: m.jac(x)[i]})
        if g["hi"].size: i=g["hi"]; cons.append({"type":"ineq","fun":lambda x,i=i: pn.cu[i]-m.c(x)[i],"jac":lambda x,i=i: -m.jac(x)[i]})
    bounds=[(None if not np.isfinite(lo) else float(lo), None if not np.isfinite(hi) else float(hi)) for lo,hi in zip(pn.xl,pn.xu)]
    try:
        r = mincon.minimize(m.f, pn.x0, jac=m.grad, bounds=bounds, constraints=cons or None, method="interior-point", options={"maxiter": 1, "threads": 1})
        notes = [n for n in r.notes if n.startswith("Derivative check")]
        if notes: mild.append((name, notes[0][:120]))
        else: ok += 1
    except RuntimeError as e:
        msg = str(e)
        if "disagree" in msg: gross.append((name, msg[:200]))
        else: errs.append((name, msg[:120]))
print(f"{len(man)} problems in {time.time()-t0:.0f}s: silent {ok}, mild notes {len(mild)}, GROSS false alarms {len(gross)}, other errors {len(errs)}")
for n, msg in gross: print("  GROSS", n, msg)
for n, msg in mild: print("  mild ", n, msg)
for n, msg in errs: print("  err  ", n, msg)
