"""Plotly presentation of precomputed analysis frames."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.support_resistance.models import SupportResistanceResult, VolumeProfile

UP = "#44d7b6"
DOWN = "#f67d8c"
GOLD = "#f3c87a"
BLUE = "#80aaff"
MUTED = "#8e9eb3"


def _layout(figure: go.Figure, height: int) -> go.Figure:
    figure.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Microsoft YaHei, sans-serif", color="#d5dfeb", size=12),
        margin=dict(l=8, r=8, t=28, b=8),
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=11)),
        hovermode="x unified", dragmode="pan",
    )
    figure.update_xaxes(showgrid=False, rangeslider_visible=False, fixedrange=False)
    figure.update_yaxes(gridcolor="rgba(142,158,179,0.12)", zeroline=False, fixedrange=False)
    return figure


def _price_volume(frame: pd.DataFrame, intraday: bool) -> go.Figure:
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.035, row_heights=[0.78, 0.22])
    x = frame["timestamp"].dt.strftime("%H:%M") if intraday else frame["session"].map(str)
    for completed, name in [(True, "已完成 K 线"), (False, "未完成 K 线 · 暂定")]:
        mask = frame["complete"].eq(completed)
        bars = frame.loc[mask]
        if bars.empty:
            continue
        hover = [f"{stamp}<br>开 {row.open:.2f} · 高 {row.high:.2f}<br>"
                 f"低 {row.low:.2f} · 收 {row.close:.2f}<br>{name}"
                 for stamp, row in zip(x[mask], bars.itertuples())]
        figure.add_trace(go.Candlestick(
            x=x[mask], open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"],
            name=name, increasing_line_color=UP, decreasing_line_color=DOWN,
            opacity=1 if completed else 0.4, hovertext=hover, hoverinfo="text"), row=1, col=1)
    overlays = [("vwap", "VWAP · 5 分钟估算", GOLD)] if intraday else [
        ("sma20", "SMA20", GOLD), ("sma50", "SMA50", BLUE)]
    for column, name, color in overlays:
        figure.add_trace(go.Scatter(x=x, y=frame[column], mode="lines", name=name,
                                   line=dict(color=color, width=1.8), connectgaps=False,
                                   hovertemplate="%{y:.2f}<extra>" + name + "</extra>"), row=1, col=1)
    colors = [UP if close >= opening else DOWN
              for close, opening in zip(frame["close"], frame["open"])]
    figure.add_trace(go.Bar(x=x, y=frame["volume"], marker_color=colors,
                           marker_line_width=0, opacity=0.5, name="成交量",
                           hovertemplate="%{y:,.0f} 股<extra>成交量</extra>"), row=2, col=1)
    figure.update_yaxes(title_text="USD", row=1, col=1)
    figure.update_yaxes(title_text="股", tickformat="~s", row=2, col=1)
    # Explicit category order preserves the position of provisional bars and overlays.
    figure.update_xaxes(type="category", categoryorder="array", categoryarray=x.tolist(),
                        nticks=9 if intraday else 7)
    return _layout(figure, 410 if intraday else 460)


def daily_chart(frame: pd.DataFrame, support_resistance: SupportResistanceResult | None = None,
                *, show_zones: bool = True, show_profile: bool = True,
                show_avwap: bool = True) -> go.Figure:
    """Overlay service results on the price subplot, leaving volume separate."""
    figure = _price_volume(frame, intraday=False)
    if support_resistance is None or not support_resistance.available or frame.empty:
        return figure
    if show_zones:
        selected = (*support_resistance.supports, *support_resistance.resistances,
                    *support_resistance.interactions)
        names = {"support": "支撑", "resistance": "阻力", "interaction": "互动"}
        for zone in selected:
            color = UP if zone.kind == "support" else DOWN if zone.kind == "resistance" else GOLD
            figure.add_hrect(
                y0=zone.lower, y1=zone.upper, fillcolor=color, opacity=0.10,
                line=dict(color=color, width=0.5), row=1, col=1,
                annotation_text=f"{names[zone.kind]} {zone.zone_id}",
                annotation_position="top left", annotation_font_size=10,
                annotation_font_color=color,
            )
    if show_profile and support_resistance.profile.available:
        profile = support_resistance.profile
        for label, value in (("估算 POC", profile.poc), ("估算 VAL", profile.val), ("估算 VAH", profile.vah)):
            if value is not None:
                figure.add_hline(y=value, line=dict(color=MUTED, width=0.8, dash="dot"),
                                row=1, col=1, annotation_text=label,
                                annotation_position="top right", annotation_xanchor="right",
                                annotation_xshift=-2, annotation_font_size=9,
                                annotation_font_color=MUTED)
    if show_avwap:
        visible_sessions = set(frame["session"])
        for anchor, color in zip(support_resistance.anchors, ("#b99bff", "#67c8ed")):
            points = [point for point in anchor.points if point.session in visible_sessions]
            if not points:
                continue
            figure.add_trace(go.Scatter(
                x=[str(point.session) for point in points], y=[point.value for point in points],
                name=f"AVWAP {'低点' if anchor.kind == 'low' else '高点'} · {anchor.anchor_session:%m-%d}",
                meta={"anchor_id": anchor.anchor_id}, mode="lines", connectgaps=False,
                line=dict(color=color, width=1.6, dash="dash"),
                hovertemplate="%{y:.2f}<extra>" + anchor.label + "</extra>"), row=1, col=1)
    # Off-screen historical zones must not stretch the visible candle scale.
    low, high = float(frame["low"].min()), float(frame["high"].max())
    padding = (high - low) * 0.06 if high > low else high * 0.01
    figure.update_yaxes(range=[max(0, low - padding), high + padding], row=1, col=1)
    figure.update_layout(height=510, margin=dict(l=8, r=8, t=52, b=8))
    return figure


def volume_profile_chart(profile: VolumeProfile) -> go.Figure:
    """Daily-bar estimated volume by price; no buy/sell volume split inferred."""
    colors = [GOLD if row.center == profile.poc else UP if row.node == "hvn"
              else MUTED if row.node == "lvn" else BLUE for row in profile.bins]
    figure = go.Figure(go.Bar(
        x=[row.volume for row in profile.bins], y=[row.center for row in profile.bins],
        width=[row.upper - row.lower for row in profile.bins], orientation="h",
        marker_color=colors, opacity=0.65, name="估算成交量",
        hovertemplate="价格 %{y:.2f}<br>估算成交量 %{x:,.0f}<extra></extra>"))
    figure.update_xaxes(title_text="估算成交量 · 股", tickformat="~s")
    figure.update_yaxes(title_text="USD")
    if profile.val is not None and profile.vah is not None:
        figure.add_hrect(y0=profile.val, y1=profile.vah, fillcolor=BLUE, opacity=0.05, line_width=0)
    return _layout(figure, 310)


def intraday_chart(frame: pd.DataFrame) -> go.Figure:
    return _price_volume(frame, intraday=True)


def rsi_chart(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_hrect(y0=30, y1=70, fillcolor=BLUE, opacity=0.05, line_width=0)
    for level in (30, 70):
        figure.add_hline(y=level, line_dash="dot", line_color=MUTED,
                         annotation_text=str(level), annotation_font_color=MUTED)
    completed = frame["complete"]
    figure.add_trace(go.Scatter(x=frame["session"].map(str), y=frame["rsi14"].where(completed),
                               mode="lines", line=dict(color=BLUE, width=1.8), name="RSI14",
                               connectgaps=False, hovertemplate="RSI14 %{y:.2f}<extra></extra>"))
    forming = frame.loc[~completed & frame["rsi14"].notna()]
    if not forming.empty:
        figure.add_trace(go.Scatter(x=forming["session"].map(str), y=forming["rsi14"],
                                   mode="markers", marker=dict(color=BLUE, symbol="diamond-open", size=8),
                                   name="暂定 RSI14"))
    figure.update_yaxes(range=[0, 100], tickvals=[0, 30, 50, 70, 100])
    figure.update_xaxes(type="category", categoryorder="array",
                        categoryarray=frame["session"].map(str).tolist(), nticks=7)
    return _layout(figure, 175)
