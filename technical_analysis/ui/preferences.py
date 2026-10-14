"""The reward/risk switch both pages share; the choice is saved across restarts."""
from dataclasses import replace

import streamlit as st

from analysis.strategy import StrategyConfig
from data.preferences import Preferences, load_preferences, save_preferences

_KEY = "reward_risk_gate"
_HELP = ("开启后，新开仓还需满足：上方最近前高的空间至少是到吊灯止损距离的 2 倍。"
         "默认关闭：在 15 只大盘股 6 年的回测中，这一条件过滤掉约三分之二的交易，"
         "而被过滤的交易平均表现与保留的相近。设置会保存，自选股和个股分析页共用。")


def reward_risk_gate() -> bool:
    if _KEY not in st.session_state:
        st.session_state[_KEY] = load_preferences().reward_risk_gate
    return st.session_state[_KEY]


def strategy_config() -> StrategyConfig:
    return replace(StrategyConfig(), use_reward_risk_gate=reward_risk_gate())


def _changed(widget_key: str) -> None:
    value = bool(st.session_state[widget_key])
    st.session_state[_KEY] = value
    try:
        save_preferences(Preferences(reward_risk_gate=value))
    except OSError:
        st.toast("设置未能保存到文件，仅本次打开有效。")


def reward_risk_toggle(container, widget_key: str) -> bool:
    """Render the switch; every page reads the same saved value."""
    st.session_state[widget_key] = reward_risk_gate()
    return container.toggle("买入需盈亏比 ≥ 2", key=widget_key, on_change=_changed, args=(widget_key,), help=_HELP)
