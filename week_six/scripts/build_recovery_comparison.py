"""
Recovery comparison table: no-recovery vs v2 vs v3 for SAC+HER.

Reads:  results/data/episode_results.csv
Writes: results/data/recovery_comparison.csv

Key columns
-----------
no_recovery      — mean success rate, sac_her, no recovery (B1)
recovery_v2      — mean success rate, sac_her_recovery, v2  (B2)
recovery_v3      — mean success rate, sac_her_recovery, v3  (B3)
delta_v2_none    — v2 − no_recovery (pp improvement)
delta_v3_none    — v3 − no_recovery (pp improvement)
delta_v3_v2      — v3 − v2 (incremental gain of sustained window)
c5_recovery_score — max(0, v3 − no_recovery)   [CLAUDE.md §10 formula]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import DATA_DIR, ALL_CONDITIONS


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else float("nan")


def main() -> None:
    ep_path = os.path.join(DATA_DIR, "episode_results.csv")
    df = pd.read_csv(ep_path)
    print(f"[recovery] Loaded {len(df)} episodes from {ep_path}")

    # Keep only sac_her base method (with and without recovery).
    sac_df = df[df["method"].isin(["sac_her", "sac_her_recovery_v2", "sac_her_recovery_v3"])].copy()

    rows = []
    for condition in ALL_CONDITIONS:
        cond = sac_df[sac_df["condition"] == condition]

        no_rec = _safe_mean(cond[cond["method"] == "sac_her"]["success"])
        v2     = _safe_mean(cond[cond["method"] == "sac_her_recovery_v2"]["success"])
        v3     = _safe_mean(cond[cond["method"] == "sac_her_recovery_v3"]["success"])

        def _delta(a: float, b: float) -> float:
            return round(a - b, 3) if not (np.isnan(a) or np.isnan(b)) else float("nan")

        # C5 recovery_score: max(0, v3 − no_recovery)  — §10 of CLAUDE.md.
        c5 = max(0.0, v3 - no_rec) if not (np.isnan(v3) or np.isnan(no_rec)) else float("nan")

        rows.append({
            "condition":         condition,
            "no_recovery":       round(no_rec, 3) if not np.isnan(no_rec) else float("nan"),
            "recovery_v2":       round(v2,     3) if not np.isnan(v2)     else float("nan"),
            "recovery_v3":       round(v3,     3) if not np.isnan(v3)     else float("nan"),
            "delta_v2_none":     _delta(v2, no_rec),
            "delta_v3_none":     _delta(v3, no_rec),
            "delta_v3_v2":       _delta(v3, v2),
            "c5_recovery_score": round(c5, 3) if not np.isnan(c5) else float("nan"),
        })

    out_df = pd.DataFrame(rows)

    out_path = os.path.join(DATA_DIR, "recovery_comparison.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[recovery] Wrote {len(out_df)} rows → {out_path}")

    print("\n=== Recovery Comparison: SAC+HER Success Rate per Condition ===\n")
    print(out_df.to_string(index=False))

    # Summary stats across all attacked conditions (exclude clean).
    attacked = out_df[out_df["condition"] != "clean"]
    print("\n--- Mean across all 7 attacked conditions ---")
    for col in ["no_recovery", "recovery_v2", "recovery_v3",
                "delta_v2_none", "delta_v3_none", "c5_recovery_score"]:
        if col in attacked.columns:
            print(f"  {col:25s}: {attacked[col].mean():.3f}")


if __name__ == "__main__":
    main()
