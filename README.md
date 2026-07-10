# TAIRO Trustworthy AI Robotics

## Gym Robot Playground (`gym_app.py`)

Interactive Streamlit app for driving Gymnasium/MuJoCo robot simulations —
including the `gymnasium-robotics` Fetch arm tasks (Reach, Push, Pick & Place,
Slide) and the classic locomotion envs (Ant, Humanoid, HalfCheetah, ...).

### Run

```bash
pip install -r requirements.txt
streamlit run gym_app.py
```

### Features

- Environment picker with seed control and episode reset
- Three action sources: per-actuator manual sliders, random, or zero/hold
- Single-step or live animated playback (`Play (live)`)
- Episode recording to inline MP4 video
- Live metrics (step, reward, termination/truncation) and observation inspector
  (handles the Dict observations of the Fetch envs)

### macOS note

MuJoCo offscreen rendering off the main thread is fragile on macOS:
gymnasium's glfw renderer SIGTRAPs the process, and a CGL `mujoco.Renderer`
used across threads deadlocks. The app therefore routes all renderer
operations through a single dedicated render thread — see `_render_pool()` in
`gym_app.py` before touching the rendering code.
