"""Streamlit entry point: a watchlist home page and the single-ticker detail page."""
import streamlit as st

from ui.theme import apply_theme

st.set_page_config(page_title="价格观察室 · 股票技术分析", page_icon="◈", layout="wide")
apply_theme()
navigation = st.navigation([
    st.Page("views/home.py", title="自选股总览", icon=":material/view_list:", default=True, url_path="watchlist"),
    st.Page("views/detail.py", title="个股分析", icon=":material/show_chart:", url_path="analysis"),
], position="top")
navigation.run()
