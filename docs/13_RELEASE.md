# Initial Python release

The first public version is **0.1.0, experimental/Alpha**. It is not a completed
replacement for fmincon. The packaged README is the public description; keep
its supported platforms, simple examples and limitations accurate.

The public entry points are `minimize` (SciPy inequality signs) and `fmincon`
(MATLAB constraint inputs/signs). Users supply a function, starting point and
constraints; derivatives and options are optional. The result has `.x`, `.fun`,
`.success`, `.maxcv`, `.message` and diagnostics. Do not imply MATLAB output
tuple or MATLAB options compatibility.

## Release procedure

1. Run the workspace tests, release fixture gate and Clippy in the documented
   order. Run Python tests against installed wheels, not a source-path import.
2. Regenerate `scripts/build_license_notice.py` when Cargo dependencies change.
   Include both project licenses and the bundled Rust dependency notices.
3. Build from a clean commit/archive. Do not include the user's uncommitted
   edits or local caches. Build an sdist and rebuild wheels from its contents
   to prove all workspace path dependencies are included.
4. Validate all upload files with `python -m twine check --strict`. Check their
   contents for the licenses, correct metadata and absence of local artifacts.
5. Upload exactly the reviewed artifacts. PyPI authentication must belong to
   the project owner. Never place tokens in the repository, logs or chat.
6. Confirm the release via PyPI, then install it from PyPI in a fresh environment
   and rerun the documented example before claiming public availability.

For direct local publication, configure a token in the standard external
`~/.pypirc` file (username `__token__`) or a local credential store. Then use:

```text
python -m twine upload --non-interactive --repository-url https://upload.pypi.org/legacy/ <reviewed distribution files>
```

No PyPI account/token was configured at the beginning of this preparation.
The package page returned 404; that does not guarantee PyPI will accept the
name. An actual authenticated upload decides ownership/name eligibility.

Alternatively, `.github/workflows/publish.yml` builds, installs and tests
Windows/Linux wheels and rebuilds/tests an sdist. It publishes only on a manual
dispatch with `publish=true`, after configuring a PyPI Trusted Publisher for
the real repository, workflow filename `publish.yml`, and environment `pypi`.
There is no Git remote configured yet; this workflow has not run on GitHub.
Do not claim workflow success from local validation alone.

Sources: [maturin distribution guide](https://www.maturin.rs/distribution.html),
[PyPA publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/),
[PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## Support scope

The dependency manifests require Rust 1.83; the previous 1.80 claim was below
PyO3/NumPy's requirement. `cargo +1.83.0 check --workspace --locked` was verified
on Windows. Native kernels and packaging are tested locally on Windows x86_64
and Ubuntu under WSL x86_64, standard CPython 3.12. The Linux build uses zig and
maturin's manylinux2014 audit. The abi3 tag permits CPython 3.9+; tests on every
minor version, macOS wheels and ARM wheels remain outstanding.

Broader solver qualification, hard per-callback budgets, SQP, sparse Python
Jacobians and a failed/empty derivative-check audit remain roadmap work. Publish
the scope honestly; do not describe all 54 fixture passes as strict convergence.
