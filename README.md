# TAIRO Trustworthy AI Robotics

Interactive webapp for the TAIRO research: trained SAC+HER manipulation
policies under adversarial sensor/actuator attacks, with rule-based recovery
and safety scoring — plus a free-play robot sandbox.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pages

### ⚔️ Attack Lab
Run the week-6 trained policies (`week_six/results/models/`) on
FetchReach-v4 / FetchPickAndPlace-v4 under any benchmark attack condition —
sensor dropout/bias, goal spoofing, object-pose spoofing, contact dropout,
action clipping/delay/reversal, gripper falsification — at a tunable
magnitude, with optional recovery (C5 v2/v3). Renders the rollout live,
side-by-side clean vs attacked on the same seed, and charts distance-to-goal,
the C4 split jerk safety metric, and intended-vs-executed action norms.

All attack/recovery/safety logic is imported directly from `week_six/` —
the app orchestrates, it does not reimplement.

### 📊 Benchmark Dashboard
Explores `week_six/results/data/`: success rate and trustworthiness scores
by condition and method (B0–B3 layers), per-component score breakdown, and
the published figures.

### 🤖 Playground
Free-play sandbox for any Gymnasium/MuJoCo robot (Fetch arms + locomotion
envs): manual per-actuator sliders, random or zero actions, live playback,
MP4 episode recording.

## Layout

```
app.py               entry point (st.navigation)
apps/                one file per page
tairo_core/render.py thread-safe MuJoCo offscreen rendering (see note below)
tairo_core/research.py  bridge to week_six: model loading + episode stepper
week_six/            research code + trained models + results (from the
                     week-6-pickandplace branch; phase1_jerk_raw.csv and the
                     2M train log stay on that branch — too big for main)
```

## macOS rendering note

MuJoCo offscreen rendering off the main thread is fragile on macOS:
gymnasium's glfw renderer SIGTRAPs the process, and a CGL `mujoco.Renderer`
used across threads deadlocks. All renderer operations are therefore routed
through a single dedicated render thread — see `tairo_core/render.py` before
touching rendering code.
