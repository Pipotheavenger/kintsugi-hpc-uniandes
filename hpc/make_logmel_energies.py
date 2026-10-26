"""Regenerate hpc/configs/logmel_energies.npy from kirad.constants.IDEAL_LOGMEL_ENERGIES.

DAM 2/3 configs need `ideal_logmel_energies: <path to .npy>` (a private Kintsugi file not in the repo).
DAM 1 applies the same per-band profile from the hard-coded constant inside its GPU feature extractor.
This script copies that constant into a .npy so DAM 2/3 normalise exactly like DAM 1.
Reads the constant from source with `ast`, so it needs only numpy (no torch / kirad import).
See hpc/BITACORA.md, step 9.

Usage (from the repo root):  python hpc/make_logmel_energies.py
"""

import ast
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kirad" / "constants.py"
OUT = ROOT / "hpc" / "configs" / "logmel_energies.npy"


def read_constant() -> np.ndarray:
    tree = ast.parse(SRC.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "IDEAL_LOGMEL_ENERGIES":
            return np.array(ast.literal_eval(node.value.args[0]), dtype=np.float32)
    raise SystemExit(f"IDEAL_LOGMEL_ENERGIES not found in {SRC}")


def main():
    arr = read_constant()
    assert arr.shape == (80,), arr.shape  # one value per Whisper log-mel band
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, arr)
    back = np.load(OUT)
    assert back.dtype == np.float32 and np.array_equal(back, arr)
    print(f"wrote {OUT}  shape={back.shape}  min={back.min():.4f}  max={back.max():.4f}  mean={back.mean():.4f}")


if __name__ == "__main__":
    main()
