"""
Checkpoint-integrity guard for the four PickAndPlace benchmark models.

Origin: the seed-independence-fix audit (2026-07-13) found that
MODEL_PATH_PICKANDPLACE and MODEL_PATH_PICKANDPLACE_RANDOMIZED silently
resolved to files byte-identical to the _500k checkpoints, while
phase1_jerk_diagnostic.py's local MODELS dict used those same constants
under a "clean_2M"/"randomized_2M" label — so two of the four "models" it
evaluated were actually duplicate 500k weights under the wrong name. This
script exists to make that exact failure mode (two of the four benchmark
checkpoint constants resolving to one file) impossible to reintroduce
silently.

Usage:
    /opt/miniconda3/envs/reu_robotics/bin/python3 scripts/verify_checkpoint_integrity.py

Exits non-zero and prints which constants collide if any two of the four
resolved checkpoint files are byte-identical (by MD5) or missing.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_PATH_PICKANDPLACE,
    MODEL_PATH_PICKANDPLACE_RANDOMIZED,
    MODEL_PATH_PICKANDPLACE_2M,
    MODEL_PATH_PICKANDPLACE_RANDOMIZED_2M,
)

CHECKPOINTS = {
    "MODEL_PATH_PICKANDPLACE (clean_500k)":            MODEL_PATH_PICKANDPLACE,
    "MODEL_PATH_PICKANDPLACE_RANDOMIZED (randomized_500k)": MODEL_PATH_PICKANDPLACE_RANDOMIZED,
    "MODEL_PATH_PICKANDPLACE_2M (clean_2M)":            MODEL_PATH_PICKANDPLACE_2M,
    "MODEL_PATH_PICKANDPLACE_RANDOMIZED_2M (randomized_2M)": MODEL_PATH_PICKANDPLACE_RANDOMIZED_2M,
}


def _resolve_zip_path(path: str) -> str:
    if os.path.exists(path):
        return path
    if os.path.exists(path + ".zip"):
        return path + ".zip"
    return path  # will fail the exists check below with a clear message


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    resolved = {}
    missing = []
    for name, path in CHECKPOINTS.items():
        actual_path = _resolve_zip_path(path)
        if not os.path.exists(actual_path):
            missing.append((name, path))
            continue
        resolved[name] = (actual_path, _md5(actual_path))

    print("=" * 78)
    print("CHECKPOINT INTEGRITY CHECK — 4 PickAndPlace benchmark models")
    print("=" * 78)
    for name, (path, md5) in resolved.items():
        print(f"  {name:<55} {md5}  {path}")

    ok = True

    if missing:
        ok = False
        print("\nMISSING FILES:")
        for name, path in missing:
            print(f"  {name}: expected at {path} (or {path}.zip)")

    # Collision check: any two distinct constants resolving to the same MD5.
    md5_to_names = {}
    for name, (path, md5) in resolved.items():
        md5_to_names.setdefault(md5, []).append(name)

    collisions = {md5: names for md5, names in md5_to_names.items() if len(names) > 1}
    if collisions:
        ok = False
        print("\nCOLLISIONS DETECTED — two or more constants resolve to the same file:")
        for md5, names in collisions.items():
            print(f"  MD5 {md5}:")
            for n in names:
                print(f"    - {n}")

    print()
    if ok:
        print("PASS — all four checkpoints present and mutually distinct.")
        return 0
    else:
        print("FAIL — see above. Do not trust benchmark results built on these "
              "checkpoints until resolved.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
