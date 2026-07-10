"""
Benchmark Dashboard — explore the week_six evaluation results:
per-condition success rates, trustworthiness scores, and the published
figures, straight from results/data/*.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from tairo_core.research import DATA_DIR, FIGURES_DIR

st.title("📊 Benchmark Dashboard")
st.caption(
    "Week 6 evaluation results: SAC+HER on FetchPickAndPlace-v4 under 11 "
    "attack conditions, benchmark layers B0–B3 (clean → attacked → +recovery)."
)


@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "summary.csv")


@st.cache_data
def load_episodes() -> pd.DataFrame:
    frames = []
    for f in sorted(Path(DATA_DIR).glob("episode_results_*.csv")):
        frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


summary = load_summary()
episodes = load_episodes()

# --- Filters -------------------------------------------------------------------
st.sidebar.title("📊 Dashboard")
layers = st.sidebar.multiselect(
    "Benchmark layer", sorted(summary["benchmark_layer"].unique()),
    default=sorted(summary["benchmark_layer"].unique()),
    help="B0 clean · B1 attacked · B2 +recovery_v2 · B3 +recovery_v3",
)
methods = st.sidebar.multiselect(
    "Method", sorted(summary["method"].unique()),
    default=sorted(summary["method"].unique()),
)
view = summary[summary["benchmark_layer"].isin(layers) & summary["method"].isin(methods)]

# --- Headline: success rate by condition ----------------------------------------
st.subheader("Success rate by attack condition")
pivot = view.pivot_table(
    index="condition", columns="method", values="success_rate", aggfunc="mean"
)
st.bar_chart(pivot, x_label="condition", y_label="success rate", horizontal=True)

# --- Trustworthiness scores -------------------------------------------------------
st.subheader("Trustworthiness score (weighted)")
tw = view.pivot_table(
    index="condition", columns="method",
    values="trustworthiness_score_weighted", aggfunc="mean",
)
st.bar_chart(tw, x_label="condition", y_label="score", horizontal=True)

# --- Component scores for a chosen condition ---------------------------------------
st.subheader("Score components by condition")
cond = st.selectbox("Condition", sorted(view["condition"].unique()))
comp_cols = [
    "reliability_score", "robustness_score", "cyber_resilience_score",
    "safety_score", "recovery_score",
]
comp = (
    view[view["condition"] == cond]
    .set_index("method")[comp_cols]
    .rename(columns=lambda c: c.replace("_score", ""))
)
st.bar_chart(comp.T, x_label="component", y_label="score")

# --- Episode-level distributions -----------------------------------------------------
if len(episodes):
    st.subheader("Episode-level final distance")
    ep_cond = episodes[episodes["condition"] == cond]
    if len(ep_cond):
        by_model = ep_cond.groupby(["env", "seed"])["final_distance"].mean().reset_index()
        st.scatter_chart(
            by_model, x="seed", y="final_distance", color="env",
            x_label="seed", y_label="mean final distance (m)",
        )

# --- Published figures ----------------------------------------------------------------
figs = sorted(Path(FIGURES_DIR).glob("*.png"))
if figs:
    st.subheader("Published figures")
    for f in figs:
        st.image(str(f), caption=f.stem, width="stretch")

# --- Raw data ---------------------------------------------------------------------------
with st.expander("Raw summary table"):
    st.dataframe(view, width="stretch")
