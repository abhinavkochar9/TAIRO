---
name: verify
description: Build/launch/drive recipe for verifying the Streamlit gymnasium robot playground (gym_app.py)
---

# Verifying gym_app.py (Streamlit + Gymnasium/MuJoCo)

## Launch
- Preferred: `preview_start` with the `gym-app` config in `.claude/launch.json`
  (uses absolute path `/opt/anaconda3/bin/streamlit` — bare `streamlit` is not
  on the harness PATH).
- Manual fallback: `/opt/anaconda3/bin/streamlit run gym_app.py --server.headless true --server.port 8501 --server.address localhost`
  — needs `dangerouslyDisableSandbox: true`; the sandbox SIGTRAPs the network bind.
- Server is up when `curl -s http://localhost:8501` returns 200 (~2s).

## Gotchas (macOS rendering — the app works around these; don't regress them)
1. gymnasium's built-in renderer (`render_mode="rgb_array"`) uses glfw, which
   requires the macOS **main thread**. Streamlit scripts run on worker threads
   → the whole server dies with SIGTRAP, silently (no traceback, exit 133).
2. `MUJOCO_GL=cgl` is rejected by gymnasium (only glfw/egl/osmesa).
3. `mujoco.Renderer` (CGL on macOS) works off-main-thread, BUT a renderer
   created on one thread **hangs the entire process (GIL held)** if used from
   another. Streamlit gives every rerun a different thread, so all renderer
   ops (create/render/close) must go through the single-thread executor
   (`_render_pool` / `_on_render_thread` in gym_app.py).

## Drive
- The sidebar starts collapsed at narrow widths — expand it first
  (`[data-testid="stExpandSidebarButton"]`) before clicking sidebar widgets;
  clicks on hidden sidebar elements land on main-area buttons.
- Buttons/radios have no stable ids; find by innerText via preview_eval.
- Useful checks: metrics via `[data-testid="stMetric"]`, errors via
  `[data-testid="stException"]`, running state via `[data-testid="stStatusWidget"]`.
- Flow worth driving: Random mode → "Run steps" → Step metric increments;
  "Reset episode" → metrics zero; "Record episode" → a `<video>` element with
  nonzero duration appears; switch env in sidebar → new robot renders.
- A "Run steps" of 50 takes ~5-10s (50 physics steps + 50 offscreen renders).
