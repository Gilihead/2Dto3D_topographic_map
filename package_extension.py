"""
Creates a distributable zip of the extension.
Run:  python package_extension.py
Output: topo3d_extension.zip  (in the project root)
"""
import zipfile, pathlib, sys

ROOT     = pathlib.Path(__file__).parent
EXT_DIR  = ROOT / "extension"
OUT_FILE = ROOT / "topo3d_extension.zip"

REQUIRED = [
    "manifest.json",
    "background.js",
    "index.html",
    "index.css",
    "index.js",
    "lib/plotly.min.js",
    "lib/xlsx.min.js",
    "INSTALL.txt",
]

# Check all required files exist
missing = [f for f in REQUIRED if not (EXT_DIR / f).exists()]
if missing:
    print("ERROR — missing files:")
    for f in missing:
        print(f"  extension/{f}")
    sys.exit(1)

# Build zip
with zipfile.ZipFile(OUT_FILE, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for f in REQUIRED:
        zf.write(EXT_DIR / f, f)

size_mb = OUT_FILE.stat().st_size / 1_048_576
print(f"Done ->  {OUT_FILE.name}  ({size_mb:.1f} MB)")
print()
print("Share this zip. Recipient follows INSTALL.txt inside.")
