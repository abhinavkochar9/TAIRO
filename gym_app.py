"""
Interactive Gymnasium robot simulation playground (Streamlit).

Run with:  streamlit run gym_app.py

Pick a robot environment, reset it, and drive it either manually with per-joint
sliders, with random actions, or by holding a fixed action. Step the sim one
frame at a time or roll out a whole episode and watch the recorded video.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import streamlit as st
import gymnasium as gym
import mujoco

try:
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    HAS_ROBOTICS = True
except ImportError:
    HAS_ROBOTICS = False

st.set_page_config(page_title="Gym Robot Playground", layout="wide")

# --- Environments to expose. All render to rgb_array and use continuous actions.
ENVS = {
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

# Fetch robot arm tasks from gymnasium-robotics (Dense = shaped rewards, nicer
# to watch than the sparse -1/0 variants).
if HAS_ROBOTICS:
    ENVS = {
        "Fetch Arm — Reach": "FetchReachDense-v4",
        "Fetch Arm — Push": "FetchPushDense-v4",
        "Fetch Arm — Pick & Place": "FetchPickAndPlaceDense-v4",
        "Fetch Arm — Slide": "FetchSlideDense-v4",
        **ENVS,
    }


def flat_obs(obs):
    """Flatten an observation to a 1-D array (Fetch envs use Dict obs)."""
    if isinstance(obs, dict):
        return np.concatenate([np.ravel(v) for v in obs.values()])
    return np.ravel(obs)


def make_env(env_id: str):
    """Create a fresh env (no gymnasium renderer — see render_frame)."""
    return gym.make(env_id)


@st.cache_resource
def _render_pool():
    """Single thread that owns ALL mujoco.Renderer operations.

    Streamlit runs each script rerun on a different worker thread. On macOS,
    gymnasium's glfw renderer SIGTRAPs off the main thread, and a CGL context
    (mujoco.Renderer) created on one thread hangs the whole process (GIL held)
    if used from another. Pinning creation, render, and close to one dedicated
    thread avoids both failure modes.
    """
    return ThreadPoolExecutor(max_workers=1)


def _on_render_thread(fn):
    return _render_pool().submit(fn).result()


def make_renderer(env):
    """Offscreen renderer + tracking camera for the env's model."""
    model = env.unwrapped.model
    renderer = _on_render_thread(
        lambda: mujoco.Renderer(model, height=480, width=480)
    )
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1 if model.nbody > 1 else 0
    cam.distance = max(model.stat.extent * 1.5, 1.0)
    cam.elevation = -20
    return renderer, cam


def render_with(renderer, cam, data):
    """Render a sim state to an RGB array on the render thread."""

    def _render():
        renderer.update_scene(data, camera=cam)
        return renderer.render()

    return _on_render_thread(_render)


def render_frame():
    """Render the current session env's state."""
    renderer, cam = st.session_state.renderer
    return render_with(renderer, cam, st.session_state.env.unwrapped.data)


def reset_env(env_id: str, seed: int):
    """(Re)build the env and store it plus fresh episode state in session_state."""
    # Close any previous env to release the MuJoCo context.
    old = st.session_state.get("env")
    if old is not None:
        try:
            old_renderer = st.session_state.renderer[0]
            _on_render_thread(old_renderer.close)
            old.close()
        except Exception:
            pass

    env = make_env(env_id)
    obs, info = env.reset(seed=seed)
    st.session_state.env = env
    st.session_state.renderer = make_renderer(env)
    st.session_state.env_id = env_id
    st.session_state.obs = obs
    st.session_state.frame = render_frame()
    st.session_state.total_reward = 0.0
    st.session_state.step_count = 0
    st.session_state.last_reward = 0.0
    st.session_state.terminated = False
    st.session_state.truncated = False


def apply_action(action: np.ndarray):
    """Advance the sim by one step with the given action."""
    env = st.session_state.env
    action = np.clip(action, env.action_space.low, env.action_space.high).astype(
        env.action_space.dtype
    )
    obs, reward, terminated, truncated, info = env.step(action)
    st.session_state.obs = obs
    st.session_state.frame = render_frame()
    st.session_state.last_reward = float(reward)
    st.session_state.total_reward += float(reward)
    st.session_state.step_count += 1
    st.session_state.terminated = bool(terminated)
    st.session_state.truncated = bool(truncated)


# --- Sidebar: environment selection & reset -------------------------------
st.sidebar.title("🤖 Robot Playground")

env_name = st.sidebar.selectbox("Environment", list(ENVS.keys()), index=0)
env_id = ENVS[env_name]
seed = st.sidebar.number_input("Seed", value=0, step=1)

# Build the env on first load or when the selection changes.
if "env" not in st.session_state or st.session_state.get("env_id") != env_id:
    reset_env(env_id, int(seed))

if st.sidebar.button("🔄 Reset episode", use_container_width=True):
    reset_env(env_id, int(seed))

env = st.session_state.env
act_dim = int(np.prod(env.action_space.shape))
act_low = env.action_space.low
act_high = env.action_space.high

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**{env_id}**\n\n"
    f"- obs dim: `{flat_obs(st.session_state.obs).size}`\n"
    f"- action dim: `{act_dim}`"
)

# --- Main layout ----------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader(f"{env_name}")
    frame_slot = st.empty()
    frame_slot.image(st.session_state.frame, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Step", st.session_state.step_count)
    m2.metric("Last reward", f"{st.session_state.last_reward:.3f}")
    m3.metric("Total reward", f"{st.session_state.total_reward:.2f}")

    if st.session_state.terminated or st.session_state.truncated:
        why = "terminated" if st.session_state.terminated else "truncated (time limit)"
        st.warning(f"Episode {why}. Reset to continue.")

with right:
    st.subheader("Controls")
    mode = st.radio(
        "Action source",
        ["Manual sliders", "Random", "Zero / hold"],
        horizontal=True,
    )

    # Build the action according to the chosen mode.
    if mode == "Manual sliders":
        st.caption("Set each actuator, then Step.")
        action = np.zeros(act_dim, dtype=np.float32)
        # Guard against huge action spaces flooding the UI with sliders.
        for i in range(act_dim):
            lo = float(act_low.flat[i]) if np.isfinite(act_low.flat[i]) else -1.0
            hi = float(act_high.flat[i]) if np.isfinite(act_high.flat[i]) else 1.0
            action[i] = st.slider(
                f"a[{i}]", min_value=lo, max_value=hi, value=0.0, step=0.01,
                key=f"slider_{env_id}_{i}",
            )
    elif mode == "Random":
        action = env.action_space.sample()
    else:  # Zero / hold
        action = np.zeros(act_dim, dtype=np.float32)

    c1, c2 = st.columns(2)
    if c1.button("▶️ Step", use_container_width=True):
        apply_action(np.asarray(action, dtype=np.float32))
        st.rerun()

    n_steps = st.number_input("Steps to run", min_value=1, max_value=1000, value=50)
    if c2.button("▶️ Play (live)", use_container_width=True):
        # Animate in place: update the frame placeholder every step so the
        # robot visibly moves, then rerun once at the end to refresh metrics.
        for _ in range(int(n_steps)):
            if st.session_state.terminated or st.session_state.truncated:
                break
            a = env.action_space.sample() if mode == "Random" else np.asarray(
                action, dtype=np.float32
            )
            apply_action(a)
            frame_slot.image(st.session_state.frame, use_container_width=True)
            time.sleep(0.02)
        st.rerun()

    st.markdown("---")
    st.caption("Record a full rollout to video.")
    ep_len = st.number_input("Max episode length", min_value=10, max_value=2000, value=300)
    rollout_mode = st.radio(
        "Rollout policy", ["Random", "Zero"], horizontal=True, key="rollout_mode"
    )
    if st.button("🎬 Record episode", use_container_width=True):
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
        _on_render_thread(rec_renderer.close)
        rec_env.close()

        # Encode to mp4 via imageio and hand the bytes to Streamlit.
        import imageio.v2 as imageio
        import io

        buf = io.BytesIO()
        imageio.mimwrite(buf, frames, format="mp4", fps=30)
        st.session_state.video = buf.getvalue()
        prog.empty()

    if st.session_state.get("video"):
        st.video(st.session_state.video)

# --- Observation inspector ------------------------------------------------
with st.expander("Observation vector"):
    obs = flat_obs(st.session_state.obs).astype(float)
    st.line_chart(obs)
    st.write(obs)
