"""Bundle dependency license notices into binary distributions, offline."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
metadata = json.loads(subprocess.check_output(
    ["cargo", "metadata", "--format-version", "1", "--locked", "--offline"], cwd=ROOT))
workspace = set(metadata["workspace_members"])
sections = ["Third-party Rust dependency notices for mincon\n"
            "Includes build-time dependencies as well as linked dependencies.\n"]
for package in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
    if package["id"] in workspace:
        continue
    folder = Path(package["manifest_path"]).parent
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.name.upper().startswith(("LICENSE", "LICENCE", "COPYING")))
    if not files:
        raise RuntimeError(f"No license notice found for {package['name']}")
    sections.append(f"\n{'='*72}\n{package['name']} {package['version']}\n"
                    f"Declared license: {package['license']}\n")
    for path in files:
        sections.append(f"\n--- {path.name} ---\n{path.read_text(encoding='utf-8')}\n")
# Not a Cargo dependency, so `cargo metadata` does not list them: the standard library,
# which is statically linked into the extension, and source code adapted into mincon itself.
MIT_TEXT = """Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
sections.append(f"\n{'='*72}\nRust standard library (statically linked into the extension module)\n"
                "Declared license: MIT OR Apache-2.0 (the Apache-2.0 text is reproduced above)\n"
                "Copyright: The Rust Project Developers (see https://thanks.rust-lang.org)\n"
                "Full notices: https://github.com/rust-lang/rust/blob/master/COPYRIGHT\n"
                f"\n--- MIT ---\nCopyright (c) The Rust Project Developers\n\n{MIT_TEXT}")
sections.append(f"\n{'='*72}\nQDLDL (https://github.com/osqp/qdldl), adapted in crates/mincon-linalg/src/ldlt.rs\n"
                "Declared license: Apache-2.0 (the text is reproduced above)\n"
                "Copyright 2018, Paul Goulart, Bartolomeo Stellato, Goran Banjac, The OSQP developers\n"
                "The elimination-tree and column-count loop and the structure of the numeric up-looking\n"
                "kernel follow QDLDL; mincon translated them to Rust and changed them (fill-reducing\n"
                "permutation, pivot-sign expectations, inertia reporting, dynamic regularization).\n")
destination = ROOT / "crates/mincon-py/python/mincon/THIRD_PARTY_LICENSES.txt"
destination.write_text("\n".join(sections), encoding="utf-8")
print(f"Wrote notices for {len(metadata['packages']) - len(workspace)} dependencies")
