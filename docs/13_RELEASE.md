# Initial Python release

The first public version is **0.1.0, experimental/Alpha**. It is not a completed
replacement for fmincon. The packaged README is the public description; keep
its supported platforms, simple examples and limitations accurate.

Published September 7, 2026: [mincon 0.1.0](https://pypi.org/project/mincon/0.1.0/).
All three public artifact hashes match the reviewed manifest. A fresh Windows
installation from PyPI passed the documented example and all 18 Python tests.

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

The initial upload lacked credentials. Publication subsequently succeeded
using the owner's external token configuration; no credentials are in the repository.

Alternatively, `.github/workflows/publish.yml` builds, installs and tests
Windows, Linux and macOS (universal2) wheels and rebuilds/tests an sdist. It
publishes only on a manual dispatch with `publish=true`, after configuring a
PyPI Trusted Publisher for the real repository, workflow filename
`publish.yml`, and environment `pypi`. A push to `master` that changes the
workflow file rehearses every build and test leg with the publish job skipped
(`inputs.publish` is empty on a push), so the workflow is never first run on
the day it has to upload.
The public repository is https://github.com/atciamb/mincon, configured as `origin`.
Do not claim workflow success from local validation alone: read the run from
the GitHub API.

## 0.2.0 (September 18, 2026)

The route is the workflow, not a local upload: three platforms cannot be built
on one laptop. In order: the version in `Cargo.toml` (workspace, the seven
internal pins, the pin in `crates/mincon-py/Cargo.toml`) and `cargo update
--workspace` for the lock file; `CHANGELOG.md`; both READMEs; the local gates;
the wheel of record built **after** the documents, because the packaged README
is inside its METADATA; an installation of that wheel into a fresh virtual
environment with the documented example and the Python tests run against it;
commit, push, CI and the rehearsal read from the API; the owner's dispatch with
`publish=true`; then the tag `v0.2.0` on the released commit, an installation
from PyPI in a fresh environment, and only then a line here saying it is
public.

Found while preparing it, and fixed before the first run: the workflow's
`validate` job ran the interior-point fixture example bare, and that example
exits 1 on any fixture miss while the member alone stands at 55/56, so the
job would have failed before a wheel was built. It now applies the verdict
`ci.yml` applies: no false report of success, and the ratchet.

0.2.0 ships before its sealed round is scored; `docs/23`, Phase F, says why and
what the packaged README therefore does not claim.

Sources: [maturin distribution guide](https://www.maturin.rs/distribution.html),
[PyPA publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/),
[PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## Support scope

The dependency manifests require Rust 1.83; the previous 1.80 claim was below
PyO3/NumPy's requirement. `cargo +1.83.0 check --workspace --locked` was verified
on Windows. Native kernels and packaging are tested locally on Windows x86_64
and Ubuntu under WSL x86_64, standard CPython 3.12. The Linux build uses zig and
maturin's manylinux2014 audit. The abi3 tag permits CPython 3.9+; tests on every
minor version and Linux ARM wheels remain outstanding. Since 0.2.0 the macOS
wheel is built and tested on GitHub's Apple-silicon runner; its x86_64 half is
built there and not executed.

Broader solver qualification, hard per-callback budgets, SQP, sparse Python
Jacobians and a failed/empty derivative-check audit remain roadmap work. Publish
the scope honestly; do not describe all 54 fixture passes as strict convergence.
