# B0 baseline reproduction (commit 8ec00bd)

Measured September 7, 2026 on the same laptop used for the MATLAB comparisons
(13th Gen Intel i7-13700H, 14 cores / 20 threads, 16 GB, Windows 11 Home
26200, Balanced power plan, Rust stable 1.95 MSVC, Python 3.13.2) and on a
2-vCPU Linux x86_64 container (Rust 1.95, Python 3.11). Nothing in the solver
was changed; the checkout is a fresh clone of `8ec00bd`.

| Check | Windows | Linux |
|---|---:|---:|
| `cargo test --workspace --release` | 151 passed, 0 failed | 151 passed, 0 failed |
| `run_testset` independent fixture verdicts | 54/54 PASS, 0 LIES | 54/54 PASS, 0 LIES |
| strict `Optimal` returns | 40/54 | 41/54 |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean | (not run) |
| `cargo fmt --all -- --check` | clean | (not run) |
| installed **PyPI** wheel `mincon-0.1.0-cp39-abi3-win_amd64.whl`, `pytest crates/mincon-py/tests` | 18 passed | (not run) |

The one platform difference is HS5: `Acceptable` on Windows, `Optimal` on
Linux, with the same objective value. The strict-Optimal count is therefore
sensitive to last-bit floating-point differences near the termination
threshold; this is noted for the failure atlas rather than treated as a
regression. The 54/54 figure counts expected non-success diagnoses (e.g.
`LocallyInfeasible`, `Unbounded`, `Acceptable`) as passes and is not a
convergence count.

Non-Optimal statuses (Windows): HS5, HS7, HS13, HS30, HS39, HS71 `Acceptable`;
HS110, HS42 `StepTolerance`; HS11, HS31, HS37, HS43 `NumericalFailure` (all
four at points with relative objective error <= 5e-8); TORTURE_INFEASIBLE
`LocallyInfeasible`; TORTURE_UNBOUNDED `Unbounded`.

Raw files: `testset-windows.txt`, `rust-tests-windows-summary.txt`,
`python-tests-windows.txt`, `clippy-windows-tail.txt`. The Python test was run
against the wheel downloaded from PyPI (`pip download mincon==0.1.0`), not the
source checkout, in a fresh `python -m venv` with numpy 2.3.1.
