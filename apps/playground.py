"""
Free-play sandbox: drive any Gymnasium/MuJoCo robot manually, randomly, or
zero-held; single-step, live playback, and episode recording to MP4.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
import gymnasium as gym

from tairo_core.render import (
    make_renderer, render_with, close_renderer, flat_obs,
)
import tairo_core.research  # noqa: F401 — registers gymnasium_robotics envs

ENVS = {
    "Fetch Arm — Reach": "FetchReachDense-v4",
    "Fetch Arm — Push": "FetchPushDense-v4",
    "Fetch Arm — Pick & Place": "FetchPickAndPlaceDense-v4",
    "Fetch Arm — Slide": "FetchSlideDense-v4",
    "Ant": "Ant-v5",
    "Humanoid": "Humanoid-v5",
    "HalfCheetah": "HalfCheetah-v5",
    "Walker2d": "Walker2d-v5",
    "Hopper": "Hopper-v5",
    "Swimmer": "Swimmer-v5",
    "Reacher": "Reacher-v5",
    "Pusher": "Pusher-v5",
    "InvertedPendulum": "InvertedPendulum-v5",
    "InvertedDoublePendulum": "InvertedDoublePendulum-v5",
}


def make_env(env_id: str):
    return gym.make(env_id)


def render_frame():
    renderer, cam = st.session_state.pg_renderer
    return render_with(renderer, cam, st.session_state.pg_env.unwrapped.data)


def reset_env(env_id: str, seed: int):
    old = st.session_state.get("pg_env")
    if old is not None:
        try:
            close_renderer(st.session_state.pg_renderer[0])
            old.close()
        except Exception:
            pass

    env = make_env(env_id)
    obs, info = env.reset(seed=seed)
    st.session_state.pg_env = env
    st.session_state.pg_renderer = make_renderer(env)
    st.session_state.pg_env_id = env_id
    st.session_state.pg_obs = obs
    st.session_state.pg_frame = render_frame()
    st.session_state.pg_total_reward = 0.0
    st.session_state.pg_step_count = 0
    st.session_state.pg_last_reward = 0.0
    st.session_state.pg_terminated = False
    st.session_state.pg_truncated = False


def apply_action(action: np.ndarray):
    env = st.session_state.pg_env
    action = np.clip(action, env.action_space.low, env.action_space.high).astype(
        env.action_space.dtype
    )
    obs, reward, terminated, truncated, info = env.step(action)
    st.session_state.pg_obs = obs
    st.session_state.pg_frame = render_frame()
    st.session_state.pg_last_reward = float(reward)
    st.session_state.pg_total_reward += float(reward)
    st.session_state.pg_step_count += 1
    st.session_state.pg_terminated = bool(terminated)
    st.session_state.pg_truncated = bool(truncated)


# --- Sidebar ----------------------------------------------------------------
st.sidebar.title("🤖 Playground")
env_name = st.sidebar.selectbox("Environment", list(ENVS.keys()), index=0)
env_id = ENVS[env_name]
seed = st.sidebar.number_input("Seed", value=0, step=1)

if "pg_env" not in st.session_state or st.session_state.get("pg_env_id") != env_id:
    reset_env(env_id, int(seed))

if st.sidebar.button("🔄 Reset episode", width="stretch"):
    reset_env(env_id, int(seed))

env = st.session_state.pg_env
act_dim = int(np.prod(env.action_space.shape))
act_low, act_high = env.action_space.low, env.action_space.high

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**{env_id}**\n\n"
    f"- obs dim: `{flat_obs(st.session_state.pg_obs).size}`\n"
    f"- action dim: `{act_dim}`"
)

# --- Main layout -------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader(env_name)
    frame_slot = st.empty()
    frame_slot.image(st.session_state.pg_frame, width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric("Step", st.session_state.pg_step_count)
    m2.metric("Last reward", f"{st.session_state.pg_last_reward:.3f}")
    m3.metric("Total reward", f"{st.session_state.pg_total_reward:.2f}")

    if st.session_state.pg_terminated or st.session_state.pg_truncated:
        why = "terminated" if st.session_state.pg_terminated else "truncated (time limit)"
        st.warning(f"Episode {why}. Reset to continue.")

with right:
    st.subheader("Controls")
    mode = st.radio(
        "Action source", ["Manual sliders", "Random", "Zero / hold"], horizontal=True
    )

    if mode == "Manual sliders":
        st.caption("Set each actuator, then Step.")
        action = np.zeros(act_dim, dtype=np.float32)
        for i in range(act_dim):
            lo = float(act_low.flat[i]) if np.isfinite(act_low.flat[i]) else -1.0
            hi = float(act_high.flat[i]) if np.isfinite(act_high.flat[i]) else 1.0
            action[i] = st.slider(
                f"a[{i}]", min_value=lo, max_value=hi, value=0.0, step=0.01,
                key=f"pg_slider_{env_id}_{i}",
            )
    elif mode == "Random":
        action = env.action_space.sample()
    else:
        action = np.zeros(act_dim, dtype=np.float32)

    c1, c2 = st.columns(2)
    if c1.button("▶️ Step", width="stretch"):
        apply_action(np.asarray(action, dtype=np.float32))
        st.rerun()

    n_steps = st.number_input("Steps to run", min_value=1, max_value=1000, value=50)
    if c2.button("▶️ Play (live)", width="stretch"):
        for _ in range(int(n_steps)):
            if st.session_state.pg_terminated or st.session_state.pg_truncated:
                break
            a = env.action_space.sample() if mode == "Random" else np.asarray(
                action, dtype=np.float32
            )
            apply_action(a)
            frame_slot.image(st.session_state.pg_frame, width="stretch")
            time.sleep(0.02)
        st.rerun()

    st.markdown("---")
    st.caption("Record a full rollout to video.")
    ep_len = st.number_input("Max episode length", min_value=10, max_value=2000, value=300)
    rollout_mode = st.radio(
        "Rollout policy", ["Random", "Zero"], horizontal=True, key="pg_rollout_mode"
    )
    if st.button("🎬 Record episode", width="stretch"):
        rec_env = make_env(env_id)
        rec_renderer, rec_cam = make_renderer(rec_env)

        def rec_frame():
            return render_with(rec_renderer, rec_cam, rec_env.unwrapped.data)

        obs, _ = rec_env.reset(seed=int(seed))
        frames = [rec_frame()]
        prog = st.progress(0.0)
        for t in range(int(ep_len)):
            a = (
                rec_env.action_space.sample()
                if rollout_mode == "Random"
                else np.zeros(act_dim, dtype=np.float32)
            )
            obs, r, term, trunc, _ = rec_env.step(a)
            frames.append(rec_frame())
            prog.progress((t + 1) / int(ep_len))
            if term or trunc:
                break
        close_renderer(rec_renderer)
        rec_env.close()

        import imageio.v2 as imageio
        import io

        buf = io.BytesIO()
        imageio.mimwrite(buf, frames, format="mp4", fps=30)
        st.session_state.pg_video = buf.getvalue()
        prog.empty()

    if st.session_state.get("pg_video"):
        st.video(st.session_state.pg_video)

with st.expander("Observation vector"):
    obs = flat_obs(st.session_state.pg_obs).astype(float)
    st.line_chart(obs)
    st.write(obs)
