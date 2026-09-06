"""Load CUTEst problems into Python through S2MPJ.

S2MPJ (Gratton & Toint, arXiv:2407.07812, BSD-3-Clause) translates 1075 CUTEst
problems from SIF into *pure Python source*. That matters enormously here: the
traditional route to CUTEst needs a Fortran toolchain, the SIF decoder, a
compiled per-problem shared library and `pycutest`'s cache — a setup that keeps
most people from ever running the real benchmark. S2MPJ needs `pip install
numpy scipy` and a `git clone`.

The mapping is direct, which is not luck: S2MPJ exposes exactly the canonical
form this project chose.

    S2MPJ                     mincon
    ----------------------    ------------------------------
    p.n, p.m                  dims
    p.x0            (n, 1)    x0
    p.xlower/xupper (n, 1)    x_L, x_U
    p.clower/cupper (m, 1)    c_L, c_U
    p.fx(x) / p.fgx(x)        objective / gradient
    p.cx(x) / p.cJx(x)        constraints / Jacobian
    p.pbclass                 CUTEst classification string

Verified against `HS71`: `n=4, m=2`, `clower=[0,0]`, `cupper=[0,inf]`,
`f(x0)=16`, `c(x0)=[12,0]`, all matching the published problem.

# The classification string

`p.pbclass` looks like `C-COOR2-AY-4-2`. The characters that matter for
filtering:

* objective type: `N` none, `C` constant, `L` linear, `Q` quadratic,
  `S` sum of squares, `O` other
* constraint type: `U` unconstrained, `X` fixed variables only, `B` bounds only,
  `N` linear network, `L` linear, `Q` quadratic, `O` other
* the digit after that is the smoothness/derivative order
* `A`/`R`/`I` says whether the problem is academic, real or modelling
* the trailing numbers are `n` and `m`

`filter_problems` parses it so a run can be restricted to, say, "constrained,
nonlinear, at most 500 variables" without loading every problem first.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

S2MPJ_REPO = "https://github.com/GrattonToint/S2MPJ.git"
S2MPJ_RAW = "https://raw.githubusercontent.com/GrattonToint/S2MPJ/master"
DEFAULT_DIR = Path(os.environ.get("S2MPJ_DIR", Path.home() / ".cache" / "s2mpj"))


def ensure_s2mpj(directory: Path = DEFAULT_DIR, shallow: bool = True) -> Path:
    """Clone S2MPJ if it is not already present, and return its path.

    A shallow clone is about 100 MB. Set `S2MPJ_DIR` to point at an existing
    checkout instead.
    """
    directory = Path(directory)
    if (directory / "python_problems").is_dir():
        return directory
    directory.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if shallow:
        cmd += ["--depth", "1"]
    cmd += [S2MPJ_REPO, str(directory)]
    print(f"cloning S2MPJ into {directory} ...", file=sys.stderr)
    subprocess.run(cmd, check=True)
    return directory


def fetch_single(name: str, directory: Path = DEFAULT_DIR) -> Path:
    """Download one problem plus the runtime library, for a quick smoke test.

    Cloning the whole repository is the right thing for a real benchmark run,
    but it is a poor first experience; this makes `runner.py --problems HS71`
    work in a couple of seconds.
    """
    target = Path(directory) / "python_problems"
    target.mkdir(parents=True, exist_ok=True)
    lib = Path(directory) / "s2mpjlib.py"
    if not lib.exists():
        urllib.request.urlretrieve(f"{S2MPJ_RAW}/s2mpjlib.py", lib)
    path = target / f"{name}.py"
    if not path.exists():
        urllib.request.urlretrieve(f"{S2MPJ_RAW}/python_problems/{name}.py", path)
    return path


@dataclass
class BenchProblem:
    """A test problem in the canonical form, with derivative callables."""

    name: str
    n: int
    m: int
    x0: np.ndarray
    xl: np.ndarray
    xu: np.ndarray
    cl: np.ndarray
    cu: np.ndarray
    f: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    cons: Callable[[np.ndarray], np.ndarray]
    jac: Callable[[np.ndarray], np.ndarray]
    classification: str

    def violation(self, x: np.ndarray) -> float:
        """Maximum constraint violation, computed by the harness.

        Deliberately *not* taken from any solver's own report: a benchmark that
        trusts each solver's self-assessment measures reporting conventions
        rather than solutions.
        """
        x = np.asarray(x, dtype=float).ravel()
        v = 0.0
        if self.m:
            c = np.asarray(self.cons(x), dtype=float).ravel()
            if not np.all(np.isfinite(c)):
                return np.inf
            v = max(v, float(np.max(np.maximum(self.cl - c, c - self.cu), initial=0.0)))
        v = max(v, float(np.max(np.maximum(self.xl - x, x - self.xu), initial=0.0)))
        return max(v, 0.0)


def load(name: str, directory: Path = DEFAULT_DIR, params: tuple = ()) -> BenchProblem:
    """Load one S2MPJ problem by name."""
    directory = Path(directory)
    pdir = directory / "python_problems"
    if not (pdir / f"{name}.py").exists():
        fetch_single(name, directory)
    for p in (str(directory), str(pdir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    mod = importlib.import_module(name)
    cls = getattr(mod, name)
    prob = cls(*params)

    def col(a) -> np.ndarray:
        return np.asarray(a, dtype=float).ravel()

    n, m = int(prob.n), int(prob.m)

    def f(x):
        return float(prob.fx(np.asarray(x, dtype=float).reshape(-1, 1)))

    def grad(x):
        _, g = prob.fgx(np.asarray(x, dtype=float).reshape(-1, 1))
        return np.asarray(g, dtype=float).ravel()

    def cons(x):
        if m == 0:
            return np.zeros(0)
        c = prob.cx(np.asarray(x, dtype=float).reshape(-1, 1))
        return np.asarray(c, dtype=float).ravel()

    def jac(x):
        if m == 0:
            return np.zeros((0, n))
        _, J = prob.cJx(np.asarray(x, dtype=float).reshape(-1, 1))
        J = J.todense() if hasattr(J, "todense") else J
        return np.asarray(J, dtype=float).reshape(m, n)

    return BenchProblem(
        name=name,
        n=n,
        m=m,
        x0=col(prob.x0),
        xl=col(prob.xlower) if n else np.zeros(0),
        xu=col(prob.xupper) if n else np.zeros(0),
        cl=col(prob.clower) if m else np.zeros(0),
        cu=col(prob.cupper) if m else np.zeros(0),
        f=f,
        grad=grad,
        cons=cons,
        jac=jac,
        classification=str(getattr(prob, "pbclass", "")),
    )


CLASSIF = re.compile(
    r"^(?P<free>[A-Z]-)?(?P<obj>[NCLQSO])(?P<con>[UXBNLQO])(?P<reg>[RI])"
    r"(?P<deriv>[0-2])-(?P<origin>[ARM])(?P<internal>[NY])-"
    r"(?P<n>[0-9]+|V)-(?P<m>[0-9]+|V)"
)


def parse_classification(s: str) -> dict:
    """Parse a CUTEst classification string; empty dict when it does not match."""
    mt = CLASSIF.match(s.strip())
    return mt.groupdict() if mt else {}


def list_problems(directory: Path = DEFAULT_DIR) -> list[str]:
    """Every problem name available in the checkout."""
    pdir = Path(directory) / "python_problems"
    if not pdir.is_dir():
        return []
    skip = {"s2mpjlib"}
    return sorted(
        p.stem for p in pdir.glob("*.py") if p.stem not in skip and not p.stem.startswith("_")
    )


def filter_problems(
    names: Iterable[str],
    directory: Path = DEFAULT_DIR,
    constrained: bool | None = None,
    max_n: int | None = None,
    max_m: int | None = None,
    min_n: int = 0,
) -> list[str]:
    """Filter by classification without importing each problem.

    Reads only the `classification = "..."` comment line from each file, so
    filtering 1075 problems takes well under a second rather than minutes.
    """
    pdir = Path(directory) / "python_problems"
    out = []
    pat = re.compile(r'classification\s*=\s*"([^"]+)"')
    for name in names:
        path = pdir / f"{name}.py"
        if not path.exists():
            continue
        text = path.read_text(errors="ignore")
        mt = pat.search(text)
        if not mt:
            continue
        info = parse_classification(mt.group(1))
        if not info:
            continue
        try:
            n = int(info["n"])
            m = int(info["m"])
        except ValueError:
            # 'V' means the size is a parameter; skip unless explicitly asked.
            continue
        if constrained is True and m == 0:
            continue
        if constrained is False and m > 0:
            continue
        if max_n is not None and n > max_n:
            continue
        if max_m is not None and m > max_m:
            continue
        if n < min_n:
            continue
        out.append(name)
    return out


if __name__ == "__main__":
    # Smoke test: load HS71 and check it against the published values.
    p = load("HS71")
    x0 = p.x0
    assert (p.n, p.m) == (4, 2), (p.n, p.m)
    assert abs(p.f(x0) - 16.0) < 1e-12, p.f(x0)
    assert np.allclose(p.cons(x0), [12.0, 0.0]), p.cons(x0)
    xstar = np.array([1.0, 4.7429994, 3.8211503, 1.3794082])
    assert abs(p.f(xstar) - 17.0140173) < 1e-6, p.f(xstar)
    assert p.violation(xstar) < 1e-6, p.violation(xstar)
    print(f"OK  {p.name}: n={p.n} m={p.m} class={p.classification}")
    print(f"    f(x0)={p.f(x0):.6f}  f(x*)={p.f(xstar):.7f}  violation(x*)={p.violation(xstar):.2e}")
