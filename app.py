"""
TAIRO — Trustworthy AI Robotics interactive app.

    streamlit run app.py

Pages:
  Attack Lab  — trained SAC+HER policies under adversarial attacks, live
  Dashboard   — week 6 benchmark results explorer
  Playground  — free-play sandbox for any Gymnasium/MuJoCo robot
"""

import streamlit as st

st.set_page_config(page_title="TAIRO — Trustworthy AI Robotics", layout="wide")

pages = st.navigation([
    st.Page("apps/attack_lab.py", title="Attack Lab", icon="⚔️", default=True),
    st.Page("apps/dashboard.py", title="Benchmark Dashboard", icon="📊"),
    st.Page("apps/playground.py", title="Playground", icon="🤖"),
])
pages.run()
