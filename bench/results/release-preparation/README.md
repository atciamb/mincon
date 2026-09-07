# Experimental 0.1.0 release preparation

September 7, 2026. Source commit: `5e0f136`.
**Not published.** An authenticated upload remains outstanding. The attempted
noninteractive upload to the production PyPI endpoint stopped with
`NonInteractive: Credential not found for API token.` No distribution was
uploaded. No repository remote or GitHub Trusted Publisher is configured.

## User-facing behavior

`from mincon import fmincon` now accepts familiar optional linear constraints,
bounds and `nonlcon -> (c, ceq)` with MATLAB signs. A function, starting point
and constraints suffice; no derivatives or algorithm configuration are needed.
It returns an OptimizeResult and grouped multipliers. The existing `minimize`
interface retains SciPy inequality signs. Unsupported option names and invalid
bound/linear-constraint shapes are rejected instead of silently ignored.

The initial nine façade tests failed before implementation because the entry
point did not exist. All **18 Python tests** now pass against the final installed
Windows and Linux wheels, including analytical solutions and multiplier signs.
A fresh Windows virtual environment installed the wheel with `--only-binary`
and NumPy was fetched automatically; the README example returned `[0.5, 0.5]`,
objective `0.5000000010000067`, success True.

## Artifacts and validation

Reviewed upload files are in `bench/artifacts/release-candidate/final-dist/`:

- `mincon-0.1.0-cp39-abi3-win_amd64.whl`
- `mincon-0.1.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
- `mincon-0.1.0.tar.gz`

`artifacts.json` records sizes and SHA-256 hashes. Preserve these files for the
upload; do not accidentally upload the earlier development wheels in `dist/`.

The sdist was built from a Git archive of the source commit, excluding the
user's pre-existing formatting edits in `solver.rs`. A Windows build from that
sdist succeeded. Its 58 entries contain all workspace path dependencies,
both licenses and the Rust dependency notice file, with no `.git`, `.venv`,
build target, Python bytecode or pytest cache directories. The Linux wheel was
also rebuilt from the extracted sdist using maturin/zig with the manylinux2014
compatibility audit. Both wheels' Python wrappers match the source commit,
and their metadata includes the Alpha classifier, license expression and both
license files; the placeholder repository URL is removed.

`twine check --strict` passed for all three files. The Rust gate remains
**151 tests, 54/54 fixture outcomes, 40 strict Optimal**, with Clippy clean.
The actual dependency minimum, Rust **1.83**, passed a locked workspace check;
the previous 1.80 metadata was incorrect. Rust dependency notices for 33
packages are bundled, including build dependencies. No dependency version or
solver tolerance changed.

Windows verification: CPython 3.12, Rust 1.95.0; fresh install NumPy 2.5.3.
Linux verification: Ubuntu/WSL x86_64, CPython 3.12.3, Rust 1.98.1,
NumPy 2.5.3, maturin 1.15.0, zig 0.16.0. The Linux linker emitted a deprecated
optimization-setting warning; the build and compatibility audit succeeded.
macOS/ARM wheels and tests across every supported Python minor remain pending.

## Finishing publication

Configure a PyPI token outside the repository/chat, or set up the real GitHub
repository and Trusted Publisher. The prepared `publish.yml` workflow tests
Windows/Linux wheels and the sdist before a manually requested publication;
it has not been executed on GitHub. See `docs/13_RELEASE.md` for instructions.

After authenticating, upload only the hash-checked final artifacts, confirm the
PyPI release, and perform a fresh `pip install mincon` from PyPI. The user has
authorized publication; renewed permission is unnecessary. The external
authentication prerequisite is the remaining immediate blocker. The package
must retain its experimental description and documented solver limitations.
