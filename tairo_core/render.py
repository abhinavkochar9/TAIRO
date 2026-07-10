"""
Thread-safe MuJoCo offscreen rendering for Streamlit.

Streamlit runs each script rerun on a different worker thread. On macOS,
gymnasium's glfw renderer SIGTRAPs off the main thread, and a CGL context
(mujoco.Renderer) created on one thread hangs the whole process (GIL held)
if used from another. Pinning creation, render, and close to one dedicated
thread avoids both failure modes.
"""

from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import streamlit as st


@st.cache_resource
def _render_pool():
    """Single thread that owns ALL mujoco.Renderer operations."""
    return ThreadPoolExecutor(max_workers=1)


def on_render_thread(fn):
    """Run fn on the dedicated render thread and return its result."""
    return _render_pool().submit(fn).result()


def make_renderer(env, height: int = 480, width: int = 480):
    """Offscreen renderer + tracking camera for the env's model."""
    model = env.unwrapped.model
    renderer = on_render_thread(
        lambda: mujoco.Renderer(model, height=height, width=width)
    )
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1 if model.nbody > 1 else 0
    cam.distance = max(model.stat.extent * 1.5, 1.0)
    cam.elevation = -20
    return renderer, cam


def render_with(renderer, cam, data) -> np.ndarray:
    """Render a sim state to an RGB array on the render thread."""

    def _render():
        renderer.update_scene(data, camera=cam)
        return renderer.render()

    return on_render_thread(_render)


def close_renderer(renderer):
    """Close a renderer on the render thread."""
    on_render_thread(renderer.close)


def flat_obs(obs) -> np.ndarray:
    """Flatten an observation to 1-D (Fetch envs use Dict obs)."""
    if isinstance(obs, dict):
        return np.concatenate([np.ravel(v) for v in obs.values()])
    return np.ravel(obs)
