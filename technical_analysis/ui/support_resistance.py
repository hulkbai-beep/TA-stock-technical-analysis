"""Presentation of calculated daily support/resistance evidence."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from analysis.support_resistance.models import SupportResistanceResult, Zone
from ui.charts import volume_profile_chart

METHODS = {"swing": "摆动价格簇", "volume_profile": "估算成交量分布", "anchored_vwap": "AVWAP 共振"}
KINDS = {"support": "支撑", "resistance": "阻力", "interaction": "价格互动"}
STATUSES = {"active": "有效", "broken": "已突破 · 待回测确认", "under_retest": "回测中"}
COMPONENTS = {"tests": "独立互动", "volume": "估算量密度", "structure": "结构共振", "reaction": "历史反应幅度", "recency": "近期相关性"}


def _value(value: float | None, decimals: int = 2) -> str:
    return "不可用" if value is None else f"{value:,.{decimals}f}"


def _time(value: datetime | None) -> str:
    return "未提供具体时刻" if value is None else value.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M %Z")


def _zone_rows(zones: tuple[Zone, ...]) -> list[dict]:
    return [{
        "区域": f"{KINDS[zone.kind]} {zone.zone_id}",
        "价格区间 · USD": f"{zone.lower:.2f}–{zone.upper:.2f}",
        "中心参考": round(zone.center, 2),
        "距快照价": f"{zone.distance_percent:+.2f}%",
        "强度 / 100": _value(zone.strength_score, 1),
        "支持方法": " / ".join(METHODS.get(method, method) for method in zone.methods),
        "回顾独立互动": zone.independent_tests,
        "近期互动": zone.recent_tests,
        "最近互动": str(zone.latest_test) if zone.latest_test else "无已确认互动",
        "状态": STATUSES[zone.status],
    } for zone in zones]


def _details(zone: Zone) -> None:
    with st.expander(f"{KINDS[zone.kind]} {zone.zone_id} · 判断依据与评分"):
        st.write(zone.explanation)
        st.caption(f"完整静态边界可知日：{zone.available_session or '不可用'}。历史触及及反应按当前边界回顾统计，不代表这些边界在当时已知。")
        for evidence in zone.evidence:
            st.write("• " + evidence)
        st.caption(f"历史平均反应：{_value(zone.reaction_atr)} ATR；估算量占比：{_value(None if zone.volume_fraction is None else zone.volume_fraction * 100)}%；估算量密度：{_value(zone.relative_volume)} 倍全窗均值。")
        if zone.role_reversed:
            st.write(f"已确认角色反转：突破日 {zone.breakout_session}，回测确认日 {zone.retest_session}。")
        elif zone.recovery_session:
            st.write(f"突破后已收回：突破日 {zone.breakout_session}，原有角色恢复确认日 {zone.recovery_session}。")
        elif zone.breakout_session:
            st.write(f"已确认突破：{zone.breakout_session}；角色反转尚未确认。")
        st.dataframe(pd.DataFrame([{
            "评分项": COMPONENTS.get(item.name, item.name),
            "归一化值": _value(item.value, 3),
            "配置权重": f"{item.configured_weight:.0%}",
            "有效权重": f"{item.effective_weight:.1%}",
            "贡献分": _value(item.contribution, 2),
            "依据": item.explanation,
        } for item in zone.score_breakdown]), hide_index=True, use_container_width=True)


def render_support_resistance(result: SupportResistanceResult, *, show_profile: bool = True) -> None:
    st.subheader("支撑与阻力 · 多方法分析")
    st.caption(f"分析窗口：最近 {result.config.lookback_sessions} 个交易日，实际 {result.actual_sessions} 根已完成日线；近期权重窗口 {result.config.recent_sessions} 个交易日。确认数据截至 {result.analysis_session or '不可用'}。")
    st.caption(f"位置与距离参考：{_value(result.reference_price)} USD · {_time(result.reference_timestamp)}{' · 暂定快照' if result.reference_provisional else ''}。突破与回测状态由已完成日线确认。")
    for warning in result.warnings:
        st.warning(warning)
    if not result.available:
        st.info(result.reason or "有效历史不足，暂无法形成可靠的支撑/阻力区间。")
        return
    displayed = (*result.supports, *result.resistances, *result.interactions)
    if displayed:
        st.dataframe(pd.DataFrame(_zone_rows(displayed)), hide_index=True, use_container_width=True)
        for zone in displayed:
            _details(zone)
    else:
        st.info(result.reason or "当前证据不足，未强制生成支撑或阻力区间。")
    st.caption("强度为解释性评分，非成功概率；排名同时考虑距离。连续触及属于同一次互动，完整反应观察窗口不足时不计算反应幅度。超出图表可视价格范围的区间可在表格中查看。")

    profile_column, anchor_column = st.columns([1.3, 1], gap="medium")
    with profile_column:
        st.markdown("**估算成交量分布（Estimated Volume Profile）**")
        if result.profile.available:
            profile = result.profile
            poc, val, vah = st.columns(3)
            poc.metric("估算 POC", _value(profile.poc))
            val.metric("估算 VAL", _value(profile.val))
            vah.metric("估算 VAH", _value(profile.vah))
            st.caption(f"实际价值区覆盖：{_value(None if profile.value_area_fraction is None else profile.value_area_fraction * 100, 1)}%；总成交量 {_value(profile.total_volume, 0)} 股。每日成交量按当日高低价范围均匀分配，非逐笔价位成交量。")
            if show_profile:
                st.plotly_chart(volume_profile_chart(profile), use_container_width=True,
                                config={"displaylogo": False})
            nodes = [{"节点": kind, "价格区间": f"{node.lower:.2f}–{node.upper:.2f}",
                      "估算量占比": f"{node.fraction:.2%}", "相对量密度": round(node.relative_density, 2)}
                     for kind, values in (("HVN", profile.hvn), ("LVN", profile.lvn)) for node in values]
            with st.expander("HVN / LVN 节点"):
                if nodes:
                    st.dataframe(pd.DataFrame(nodes), hide_index=True, use_container_width=True)
                else:
                    st.write("未检测到明显的高/低量节点。")
        else:
            st.info("估算成交量分布不可用：需要有效的正成交量数据。")
    with anchor_column:
        st.markdown("**Anchored VWAP · 日线估算**")
        if result.anchors:
            st.dataframe(pd.DataFrame([{
                "锚点": anchor.label,
                "锚点交易日": str(anchor.anchor_session),
                "可确认日": str(anchor.confirmed_session),
                "最新 AVWAP": round(anchor.latest_value, 2),
                "累计成交量": round(anchor.cumulative_volume),
            } for anchor in result.anchors]), hide_index=True, use_container_width=True)
            st.caption("从确认的主要摆动高/低点累计 HLC3×成交量；曲线在锚点确认后才显示。仅在距离静态区间足够近时计入共振。")
        else:
            st.info("尚无可用的已确认主要摆动锚点，或锚点之后的数据覆盖不完整。")
    broken = tuple(zone for zone in result.zones if zone.status == "broken")
    if broken:
        with st.expander(f"已突破区间 · {len(broken)} 个（不列入有效支撑/阻力排名）"):
            st.dataframe(pd.DataFrame(_zone_rows(broken)), hide_index=True, use_container_width=True)
    st.caption("成交量权重采用 FMP 返回值；价格按已有拆股复权口径使用。跨拆股时期成交量的调整口径仍取决于数据提供方，日线无法还原真实逐笔交易分布。")
