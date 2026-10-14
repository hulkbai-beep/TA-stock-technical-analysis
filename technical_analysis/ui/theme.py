"""Shared page styling for every dashboard view."""
import streamlit as st

_CSS = """
<style>
.block-container {padding-top:4.3rem; padding-bottom:3rem; max-width:1600px;}
h1 {font-size:2.3rem !important; letter-spacing:-.045em; margin-bottom:.2rem;}
h2 {font-size:1.22rem !important; letter-spacing:-.025em;}
h3 {font-size:1.02rem !important;}
[data-testid="stMetric"] {background:#142133; border:1px solid #24354a;
border-radius:12px; padding:18px 20px; min-height:118px;}
[data-testid="stMetricLabel"] {color:#8e9eb3; font-size:.78rem;}
[data-testid="stMetricValue"] {font-size:1.8rem;}
[data-testid="stSidebar"] {border-right:1px solid #24354a;}
.eyebrow {color:#44d7b6; letter-spacing:.13em; font-size:.74rem; font-weight:600;}
.source-strip {border-left:3px solid #44d7b6; background:#111e2d; padding:12px 16px;
border-radius:0 8px 8px 0; color:#9cacc0; font-size:.8rem; line-height:1.9; margin:14px 0 22px;}
.empty-panel {border:1px dashed #31445c; border-radius:16px; padding:52px 36px;
background:linear-gradient(130deg,#142133,#0c1420); margin-top:22px;}
.empty-panel h2 {font-size:1.5rem !important;}
.empty-panel p {color:#8e9eb3; max-width:660px; line-height:1.9;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-color:#24354a;}
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
