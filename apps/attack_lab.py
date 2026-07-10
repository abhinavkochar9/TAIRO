"""
Attack Lab — run the trained SAC+HER policies under the week_six adversarial
attack conditions, live. Single attacked rollout or clean-vs-attacked A/B on
the same seed, with per-step distance/jerk/recovery telemetry.

All attack, recovery, and safety-metric code is imported from week_six —
this page only orchestrates and visualizes it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from tairo_core.render import make_renderer, render_with, close_renderer
from tairo_core.research import (
    ENVS, RECOVERY_OPTIONS, ATTACK_LEVELS,
    SAFETY_ARM_JERK_THRESHOLD,
    load_model, make_env, run_episode_steps,
)

st.title("⚔️ Attack Lab")
st.caption(
    "Adversarial robustness, live: the TAIRO benchmark's sensor & actuator "
    "attacks applied to trained SAC+HER policies, with optional rule-based "
    "recovery (C5) and the C4 safety jerk metric."
)

# --- Sidebar configuration ----------------------------------------------------
st.sidebar.title("⚔️ Attack Lab")
env_id = st.sidebar.selectbox("Environment", list(ENVS.keys()), index=1)
env_cfg = ENVS[env_id]

policy_options = ["rule_based"] + list(env_cfg["models"].keys())
policy = st.sidebar.selectbox("Policy", policy_options, index=min(1, len(policy_options) - 1))

condition = st.sidebar.selectbox("Attack condition", env_cfg["conditions"], index=0)
default_level = float(ATTACK_LEVELS.get(condition, 0.0))
attack_level = st.sidebar.slider(
    "Attack magnitude", 0.0, 0.5, default_level, 0.01,
    help="Benchmark value pre-filled from config.ATTACK_LEVELS. "
         "Structural attacks (dropout/delay/reversal/grip-falsify) ignore it.",
)
recovery = st.sidebar.selectbox("Recovery (C5)", RECOVERY_OPTIONS, index=0)
seed = st.sidebar.number_input("Seed", value=0, step=1)
max_steps = st.sidebar.slider("Max steps", 25, env_cfg["max_steps"], 100, 25)
ab_mode = st.sidebar.toggle("A/B: clean vs attacked", value=True)

model = None
if policy != "rule_based":
    model = load_model(str(env_cfg["models"][policy]), env_id, env_cfg["max_steps"])


def run_live(conditions_to_run):
    """Run one or two episodes interleaved, animating frames side by side."""
    n = len(conditions_to_run)
    cols = st.columns(n)
    frame_slots, dist_slots = [], []
    for col, (label, _cond, _rec) in zip(cols, conditions_to_run):
        with col:
            st.subheader(label)
            frame_slots.append(st.empty())
            dist_slots.append(st.empty())

    envs, renderers, gens, logs = [], [], [], []
    for label, cond, rec in conditions_to_run:
        e = make_env(env_id, max_steps)
        renderer, cam = make_renderer(e)
        renderers.append(renderer)
        envs.append(e)
        gens.append(run_episode_steps(
            e, policy, model, cond, attack_level, rec,
            int(seed), int(max_steps),
            render_fn=lambda data, r=renderer, c=cam: render_with(r, c, data),
        ))
        logs.append([])

    done = [False] * n
    prog = st.progress(0.0, text="Rolling out…")
    t = 0
    while not all(done):
        for i, g in enumerate(gens):
            if done[i]:
                continue
            step = next(g, None)
            if step is None:
                done[i] = True
                continue
            logs[i].append({k: v for k, v in step.items() if k != "frame"})
            if step["frame"] is not None:
                frame_slots[i].image(step["frame"], width="stretch")
            dist_slots[i].caption(
                f"t={step['t']}  dist={step['distance']:.3f}  "
                f"{'✅' if step['is_success'] else ''}"
                f"{'🛟' if step['recovery_triggered'] else ''}"
                f"{'⚠️jerk' if step['safety_violation'] else ''}"
            )
        t += 1
        prog.progress(min(t / max_steps, 1.0), text=f"Rolling out… step {t}")

    prog.empty()
    for r in renderers:
        close_renderer(r)
    for e in envs:
        e.close()
    return [pd.DataFrame(l) for l in logs]


run = st.button("🚀 Run", type="primary", width="stretch")

if run:
    if ab_mode and condition != "clean":
        runs = [("Clean", "clean", "none"), (f"Attack: {condition}", condition, recovery)]
    else:
        label = "Clean" if condition == "clean" else f"Attack: {condition}"
        runs = [(label, condition, recovery)]

    dfs = run_live(runs)
    st.session_state.al_results = (runs, dfs)

# --- Results ------------------------------------------------------------------
if st.session_state.get("al_results"):
    runs, dfs = st.session_state.al_results
    st.markdown("---")
    st.subheader("Episode verdict")

    vcols = st.columns(len(runs))
    for col, (label, _c, _r), df in zip(vcols, runs, dfs):
        with col:
            success = bool(df["is_success"].iloc[-1]) if len(df) else False
            st.metric(
                label,
                "SUCCESS ✅" if success else "FAIL ❌",
                help="is_success at the final timestep (benchmark C1)",
            )
            st.metric("Final distance (m)", f"{df['distance'].iloc[-1]:.4f}")
            st.metric("Total reward", f"{df['total_reward'].iloc[-1]:.1f}")
            st.metric("Safety violations (C4)", int(df["safety_violation"].sum()))
            st.metric("Recovery steps (C5)", int(df["recovery_triggered"].sum()))

    st.subheader("Distance to goal")
    dist_df = pd.DataFrame(
        {label: df.set_index("t")["distance"] for (label, _c, _r), df in zip(runs, dfs)}
    )
    st.line_chart(dist_df, x_label="step", y_label="distance (m)")

    st.subheader("Arm jerk (C4 safety channel)")
    jerk_df = pd.DataFrame(
        {label: df.set_index("t")["arm_jerk"] for (label, _c, _r), df in zip(runs, dfs)}
    )
    jerk_df["threshold"] = SAFETY_ARM_JERK_THRESHOLD
    st.line_chart(jerk_df, x_label="step", y_label="‖Δ executed action‖")

    st.subheader("Action norms — intended vs executed")
    for (label, _c, _r), df in zip(runs, dfs):
        st.caption(label)
        st.line_chart(
            df.set_index("t")[["intended_norm", "executed_norm"]],
            x_label="step", y_label="‖action‖",
        )

    with st.expander("Step-level data"):
        for (label, _c, _r), df in zip(runs, dfs):
            st.caption(label)
            st.dataframe(df, width="stretch", height=240)
