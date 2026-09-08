# Release-candidate validation (not published)

Candidate: `research/s0-s1` branch head after the S6 fixes (C2 solver +
unbounded diagnosis). Version number left at 0.1.0 on purpose: **no PyPI
upload was made or authorized**; a release needs a version bump, changelog
and the owner's authorization.

| check | Windows (laptop) | Linux (container) |
|---|---|---|
| `cargo test --workspace --release` | 160 passed | 160 passed |
| fixture gate `run_testset` | 54/54, 50 strict Optimal | 54/54 |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean | clean |
| `cargo fmt --all -- --check` | clean | clean |
| MSRV `cargo +1.83.0 check --workspace` | passes | not run (1.95 only) |
| `maturin build --release` | `mincon-0.1.0-cp39-abi3-win_amd64.whl`, SHA-256 `341adc4d…e22bf` | `mincon-0.1.0-cp39-abi3-manylinux_2_35_x86_64.whl`, SHA-256 `511e514b…f91c9` |
| `pytest crates/mincon-py/tests` on the installed wheel | 23 passed | 23 passed |
| README example (`fmincon(... nonlcon=...)`) | — | `[0.5, 0.5]`, 0.5, success |
| `maturin sdist` contents | — | 36 `.rs` files, licences, READMEs; no private files |
| dependency set | unchanged from 0.1.0 (no new crates) | |

Compatibility notes: the Linux wheel here is tagged `manylinux_2_35`
(container glibc); the published 0.1.0 wheel was `manylinux_2_34`. A wider
tag needs a manylinux build environment. macOS/ARM not qualified. Python
≥ 3.9 abi3; tested on 3.11 (Linux) and 3.13 (Windows).

Behaviour changes since 0.1.0 that a user can observe: default
`OptimalityTolerance`-equivalent 1e-6 (was 1e-8) and `acceptable` 1e-4;
`Acceptable`/`StepTolerance` where `NumericalFailure` was reported at
feasible near-KKT points; `Unbounded` for usable exits that ran away;
fixed-variable multipliers reconstructed; `check_derivatives=True` now stops
a solve on a failed or inconclusive check; `'jac'` in constraint dicts,
`nonlcon_jac`, `ncev`/`ncjev` counts, `options={'barrier': ...}`; `nonlcon`
called once per point; plain Python calls no longer run three portfolio
members under the GIL.
