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
destination = ROOT / "crates/mincon-py/python/mincon/THIRD_PARTY_LICENSES.txt"
destination.write_text("\n".join(sections), encoding="utf-8")
print(f"Wrote notices for {len(metadata['packages']) - len(workspace)} dependencies")
