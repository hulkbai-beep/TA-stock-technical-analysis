"""Render structured technical assessments without embedding trading rules."""
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

from analysis.signals import TechnicalAssessment


def _time(value: datetime | None) -> str:
    if value is None:
        return "不可用"
    return value.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M %Z")


def render_assessment(assessment: TechnicalAssessment, holding: bool) -> None:
    """Holding status selects one precomputed guidance object only."""
    guidance = assessment.position if holding else assessment.entry
    with st.container(border=True):
        st.subheader("技术强弱")
        daily_column, intraday_column, action_column = st.columns(3)
        daily_column.metric("日线判断", assessment.daily.label)
        intraday_column.metric("盘中判断", assessment.intraday.label)
        action_column.metric("操作参考", guidance.label)
        st.caption(f"{'持仓管理' if holding else '新建仓观察'} · {guidance.timeframe}")
        st.write(guidance.explanation)
        st.markdown(f"**入场吸引力**　{assessment.entry_attractiveness}")
        st.caption(f"动量观察：{assessment.momentum_label}")
        st.caption(f"日线依据时点：{_time(assessment.daily.as_of)}　 ·　盘中依据时点：{_time(assessment.intraday.as_of)}")
        conflicts = tuple(dict.fromkeys((*assessment.daily.conflicts, *assessment.intraday.conflicts, *guidance.conflicts)))
        if conflicts:
            st.warning("冲突或限制：" + "；".join(conflicts))
        st.markdown(f"**触发状态**　{guidance.trigger_status}")
        trigger_column, invalidation_column = st.columns(2)
        with trigger_column.expander("触发条件"):
            for condition in guidance.conditions:
                st.write("• " + condition)
        with invalidation_column.expander("失效条件"):
            for condition in guidance.invalidation:
                st.write("• " + condition)
        with st.expander("判断依据"):
            st.markdown(f"**日线判断：{assessment.daily.label}**")
            st.write(assessment.daily.explanation)
            for evidence in assessment.daily.evidence:
                st.write("• " + evidence)
            st.markdown(f"**盘中判断：{assessment.intraday.label}**")
            st.write(assessment.intraday.explanation)
            for evidence in assessment.intraday.evidence:
                st.write("• " + evidence)
            st.caption(f"规则评估时间：{_time(assessment.evaluated_at)}。日线时间来自交易日历推定的收盘，5 分钟时间为区间结束；均非逐笔成交时间。")
        for note in assessment.provisional_notes:
            st.caption("暂定观察：" + note)
        st.caption("初始技术启发式规则，尚未做历史验证。均线和 VWAP 是动态计算参考，不是固定止损或目标价。")
