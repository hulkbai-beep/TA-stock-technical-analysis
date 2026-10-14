"""Render structured technical assessments without embedding trading rules."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
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
        daily_column, intraday_column, score_column, action_column = st.columns(4)
        daily_column.metric("日线判断", assessment.daily.label)
        intraday_column.metric("盘中判断", assessment.intraday.label)
        score = assessment.score
        score_column.metric("综合评分 / 100", "不可用" if score is None or score.score is None else f"{score.score:.0f} · {score.grade}",
                            help="七项指标的加权评分；权重为预设先验，尚需用回测脚本验证。新开仓需至少 60 分。")
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
        if score is not None:
            with st.expander("综合评分构成"):
                st.dataframe(pd.DataFrame([{
                    "指标": item.label, "权重": f"{item.weight:.0%}",
                    "得分 (0–1)": "不可用" if item.value is None else f"{item.value:.2f}",
                    "贡献分": "—" if item.contribution is None else f"{item.contribution:.1f}",
                    "依据": item.detail,
                } for item in score.components]), hide_index=True, use_container_width=True)
                st.caption(f"可计算权重覆盖 {score.coverage:.0%}；缺失项按比例重新分配权重，覆盖低于 50% 时不给分。")
        for note in assessment.context_notes:
            st.caption("数据说明：" + note)
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
        st.caption("技术启发式规则；请用 scripts/backtest.py 在历史数据上验证后再依赖。均线、VWAP 与吊灯止损是动态计算参考，不是保证成交的价格。")
