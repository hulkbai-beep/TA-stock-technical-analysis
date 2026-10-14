"""Simplified Chinese Streamlit UI consuming structured analysis results."""
from datetime import datetime
import html
from time import perf_counter
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from config import load_settings
from data.fmp_client import FMPClient
from data.market_store import freshness_epoch
from data.normalize import session_schedule
from models import DataError
from services.analysis_service import SRConfig, analyze, assessment_for, support_resistance_for, visible_daily
from services.ticker_requests import LatestTickerRequest
from ui.assessment import render_assessment
from ui.support_resistance import render_support_resistance
from ui.charts import daily_chart, intraday_chart, rsi_chart

st.set_page_config(page_title="价格观察室 · 股票技术分析", page_icon="◈", layout="wide")
st.markdown("""
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
""", unsafe_allow_html=True)


@st.cache_resource
def get_client() -> FMPClient:
    # Configuration is read internally, so credential strings are not cache keys.
    return FMPClient(load_settings())


def time_text(value: datetime | None) -> str:
    return value.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z") if value else "未提供"


def number_text(value: float | None, decimals: int = 2) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.{decimals}f}"


def ticker_changed() -> None:
    st.session_state["ticker_changed"] = True


def reset_refresh_clock() -> None:
    st.session_state["refresh_clock"] = perf_counter()


def publish_request() -> bool:
    """Only the main Streamlit thread publishes the currently selected generation."""
    controller = st.session_state["ticker_requests"]
    state = controller.poll()
    if (state.status not in {"ready", "error"} or
            st.session_state.get("published_request") == state.request_id):
        return False
    # Input edits can arrive before a full-script rerun processes the callback.
    selected = st.session_state.get("ticker_input", "").strip().upper()
    if state.symbol is not None and selected != state.symbol:
        return False
    if state.symbol is not None and not controller.is_current(state.request_id, state.symbol):
        return False
    st.session_state["published_request"] = state.request_id
    # Start the next interval after success or failure, including slow requests.
    reset_refresh_clock()
    if state.status == "ready":
        st.session_state["analysis_result"] = state.result
        st.session_state.pop("analysis_error", None)
    else:
        st.session_state["analysis_error"] = str(state.error)
    return True


@st.fragment(run_every=0.1)
def poll_loading() -> None:
    if publish_request():
        st.rerun()
    state = st.session_state["ticker_requests"].poll()
    if state.status == "loading":
        st.caption(f"◌ 正在分析 {state.symbol} · 读取行情并计算三种结构方法…")


def start_request(symbol: str, refresh: bool = False) -> None:
    reset_refresh_clock()
    st.session_state.pop("analysis_error", None)
    try:
        settings = load_settings()
        if not settings.api_key or settings.api_key == "your_fmp_api_key":
            raise DataError("尚未配置 FMP_API_KEY。请在本项目的 .env 文件中设置密钥后重启应用。", "missing_key")
        client = get_client()
        loader = lambda: analyze(symbol, client, refresh=refresh)
    except (DataError, ValueError) as error:
        safe_error = DataError(str(error), getattr(error, "code", "invalid_configuration"))
        def loader():
            raise safe_error
    st.session_state["ticker_requests"].submit(symbol, loader, refresh=refresh)


@st.fragment(run_every=5)
def watch_auto_refresh() -> None:
    current = st.session_state.get("analysis_result")
    interval = st.session_state.get("auto_refresh_seconds", 60)
    controller = st.session_state["ticker_requests"]
    if not interval:
        st.caption("自动刷新已关闭。可点击「刷新数据」手动更新。")
        return
    if current is None:
        st.caption("分析股票后，将在常规交易时段自动刷新。")
        return
    if (controller.current_symbol != current.symbol or
            st.session_state.get("ticker_input", "").strip().upper() != current.symbol):
        return
    if controller.poll().status == "loading":
        st.caption("◌ 正在刷新行情；完成后重新计时。")
        return
    now = datetime.now(ZoneInfo("America/New_York"))
    schedule = session_schedule(now.date(), now.date())
    market_open = (not schedule.empty and
                   schedule.iloc[0].market_open <= now < schedule.iloc[0].market_close)
    if not market_open:
        st.caption("休市期间暂停定时刷新，常规交易时段自动恢复；收盘数据仍会检查更新。")
        return
    elapsed = perf_counter() - st.session_state.setdefault("refresh_clock", perf_counter())
    if elapsed >= interval:
        # A timed refresh must fetch new data even within the five-minute cache TTL.
        start_request(current.symbol, refresh=True)
        st.rerun()
    remaining = max(1, int(interval - elapsed + 0.999))
    st.caption(f"自动刷新已开启 · 每 {interval} 秒 · 距下次刷新约 {remaining} 秒。")


@st.fragment(run_every=30)
def render_current_assessment() -> None:
    # Freshness expires even with auto-fetch off or while a request is failing.
    current = st.session_state.get("analysis_result")
    if current is not None:
        render_assessment(assessment_for(current), st.session_state.get("holding_status", False))


@st.fragment(run_every=30)
def watch_daily_close() -> None:
    """A cheap clock/cache check; only a new missing daily close starts I/O."""
    current = st.session_state.get("analysis_result")
    controller = st.session_state["ticker_requests"]
    if (not st.session_state.get("auto_refresh_seconds", 60) or current is None or
            current.market_versions is None or controller.poll().status == "loading" or
            controller.current_symbol != current.symbol or
            st.session_state.get("ticker_input", "").strip().upper() != current.symbol):
        return
    now = datetime.now(ZoneInfo("America/New_York"))
    epoch = freshness_epoch("daily", now)
    missing_close = epoch is not None and (current.confirmed_daily_session is None or
                                           current.confirmed_daily_session < epoch.date())
    if not missing_close:
        return
    last_attempt = st.session_state.get("daily_close_attempt")
    if (last_attempt is not None and last_attempt[:2] == (current.symbol, str(epoch)) and
            perf_counter() - last_attempt[2] < 30):
        return
    # If the provider is late, the market store applies a 30-second retry TTL.
    cached_daily = get_client().peek_validated_daily(current.symbol, now=now)
    if cached_daily is None or getattr(cached_daily, "version", current.market_versions[0]) != current.market_versions[0]:
        st.session_state["daily_close_attempt"] = (current.symbol, str(epoch), perf_counter())
        start_request(current.symbol)
        st.rerun()


if "ticker_requests" not in st.session_state:
    st.session_state["ticker_requests"] = LatestTickerRequest()


with st.sidebar:
    st.markdown('<p class="eyebrow">◈ 价格观察室</p>', unsafe_allow_html=True)
    st.title("股票技术分析")
    st.caption("第一版 · 价格与成交量")
    st.divider()
    st.markdown("**观察框架**")
    st.markdown("日线趋势　SMA20 / SMA50\n\n动量变化　RSI14\n\n日内位置　VWAP")
    st.divider()
    st.caption("数据来源：FMP\n\n美股常规交易时段\n\nAmerica/New_York")
    st.caption("复用已校验行情 · 新交易K线自动失效\n\n最新可用数据的延迟取决于账户权限。")
    with st.expander("配置与方法"):
        st.write("在本项目的 .env 文件中设置 FMP_API_KEY。已有兼容网关可设置 FMP_BASE_URL。")
        st.write("日线固定获取 730 个自然日，先计算指标再截取显示范围。")
        st.write("RSI14 使用 Wilder 平滑；VWAP 按每个交易日重置，由 5 分钟 OHLCV 估算。")
        st.write("仅已完成日线用于确认趋势。浅色 K 线与空心菱形表示暂定值。")

st.markdown('<div class="eyebrow">价格 · 趋势 · 动量</div>', unsafe_allow_html=True)
st.title("读懂价格，保持客观。")
st.caption("用可核验的指标观察美股；每条分析都附带实际数值与规则。")

with st.container(border=True):
    ticker_column, action_column = st.columns([5, 1], vertical_alignment="bottom")
    ticker = ticker_column.text_input("美股代码", value="NVDA", placeholder="例如 NVDA、AAPL、BRK.B", max_chars=15,
                                      key="ticker_input", on_change=ticker_changed)
    submitted = action_column.button("分析", type="primary", use_container_width=True)

changed = st.session_state.pop("ticker_changed", False)
result = st.session_state.get("analysis_result")
if (submitted or changed) and ticker.strip().upper() != st.session_state["ticker_requests"].current_symbol:
    st.session_state["holding_status"] = False
if (submitted or changed) and result and ticker.strip().upper() != result.symbol:
    st.session_state.pop("analysis_result", None)
    result = None
control_left, holding_column, control_right = st.columns([3, 2, 1], vertical_alignment="center")
range_label = control_left.radio("日线显示范围", ["3M", "6M", "1Y"], index=1, horizontal=True,
                                 help="只改变显示范围；相同日期的指标数值保持一致。")
holding = holding_column.toggle("目前持有这只股票？", value=False, key="holding_status",
                                help="仅切换新建仓或持仓管理的操作参考，不改变行情和技术强弱判断；不读取账户或成本。")
refresh = control_right.button("↻ 刷新数据", disabled=result is None, use_container_width=True)
st.selectbox("自动刷新", [0, 30, 60, 300], index=2, key="auto_refresh_seconds",
             format_func=lambda seconds: {0: "关闭", 30: "每 30 秒", 60: "每 60 秒", 300: "每 5 分钟"}[seconds],
             on_change=reset_refresh_clock,
             help="页面保持打开时，在美股常规交易时段获取最新行情。5 分钟K线仍按原周期形成；数据延迟取决于提供方。")

if submitted or changed or refresh:
    requested_symbol = result.symbol if refresh and result else ticker
    start_request(requested_symbol, refresh=refresh)
publish_request()
result = st.session_state.get("analysis_result")
if st.session_state["ticker_requests"].poll().status == "loading":
    poll_loading()
watch_auto_refresh()

if st.session_state.get("analysis_error"):
    st.error(st.session_state["analysis_error"])
    if result:
        st.warning("本次请求未成功。以下保留上次成功读取的数据，请核对读取时间。")

if result is None:
    st.markdown("""<div class="empty-panel"><div class="eyebrow">从一个代码开始</div>
    <h2>一个标的，两个时间尺度。</h2>
    <p>输入美股代码并点击「分析」，查看日线趋势、RSI14 动量与最近交易日的 5 分钟走势。
    图表展示真实行情；无权限或数据不足时会说明原因。</p>
    <p>日线 K 线 + SMA20 / SMA50　 ·　 RSI14　 ·　 5 分钟 K 线 + VWAP</p></div>""",
    unsafe_allow_html=True)
    st.stop()

ui_started = perf_counter()
snapshot = result.snapshot
confirmed = result.daily.loc[result.daily["complete"]]
last_daily = confirmed.iloc[-1] if not confirmed.empty else None
st.subheader(f"{result.symbol} · 市场快照")
price_col, rsi_col, session_col, volume_col = st.columns(4)
delta = f"{snapshot.change_percent:+.2f}%" if snapshot.change_percent is not None else None
price_col.metric("最新可用价 · USD", number_text(snapshot.price), delta=delta,
                 help="最后一根可用 K 线的收盘值，并非逐笔报价。涨跌幅相对该交易日前一交易日收盘。")
rsi_col.metric("RSI14 · 已完成日线", number_text(last_daily["rsi14"] if last_daily is not None else None))
session_col.metric("快照交易日", str(snapshot.session))
volume_col.metric("该交易日累计成交量 · 股", number_text(snapshot.volume, 0),
                  help=snapshot.volume_note + " 未做同一时刻的历史量能比较。")
state = "暂定 · 读取时 K 线尚未完成，待刷新确认" if snapshot.provisional else "已完成 K 线"
source_names = {"daily": "日线 OHLCV", "intraday": "5 分钟 OHLCV"}
source = source_names.get(snapshot.source, snapshot.source)
timestamp_note = getattr(snapshot, "timestamp_note", "")
strip = (f"行情时点：{time_text(snapshot.market_timestamp)}　 ·　{state}　 ·　来源：{source}"
         f"<br>日线读取：{time_text(result.daily_retrieved_at)}　 ·　日内读取：{time_text(result.intraday_retrieved_at)}")
if timestamp_note:
    strip += f"<br>{html.escape(timestamp_note)}"
st.markdown(f'<div class="source-strip">{strip}</div>', unsafe_allow_html=True)
if snapshot.session != datetime.now(ZoneInfo("America/New_York")).date():
    st.info(f"最近可用行情来自 {snapshot.session}；此处明确显示实际交易日。")
for warning in result.warnings:
    st.warning(warning)

render_current_assessment()

with st.expander("支撑/阻力设置", expanded=True):
    window_column, count_column = st.columns(2)
    sr_window = window_column.selectbox("结构分析窗口", ["12 个月 · 252 个交易日", "6 个月 · 126 个交易日"], key="sr_window")
    sr_count = count_column.selectbox("每侧最多显示区间", [2, 3, 5], index=1, key="sr_count")
    zone_column, profile_column, avwap_column = st.columns(3)
    show_zones = zone_column.checkbox("显示支撑/阻力区", value=True, key="show_sr_zones")
    show_profile = profile_column.checkbox("显示估算成交量分布", value=True, key="show_sr_profile")
    show_avwap = avwap_column.checkbox("显示 AVWAP", value=True, key="show_sr_avwap")
sr_config = SRConfig(lookback_sessions=252 if sr_window.startswith("12") else 126,
                     max_zones_per_side=sr_count)
sr_result = support_resistance_for(result, sr_config)

chart_column, summary_column = st.columns([3.1, 1.15], gap="large")
plot_config = {"displaylogo": False, "scrollZoom": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
with chart_column:
    st.subheader(f"日线走势 · {range_label}")
    shown_daily = visible_daily(result.daily, range_label)
    st.plotly_chart(daily_chart(shown_daily, sr_result, show_zones=show_zones,
                               show_profile=show_profile, show_avwap=show_avwap),
                    use_container_width=True, config=plot_config)
    st.caption("浅色 K 线为未完成日线，相关指标暂定。价格按拆股调整；成交量仅按数据原值显示。")
    st.subheader("RSI14 · 日线动量")
    st.plotly_chart(rsi_chart(shown_daily), use_container_width=True, config=plot_config)
    st.caption("Wilder 平滑 · 30 / 70 为观察阈值，不代表必然反转。")
    render_support_resistance(sr_result, show_profile=show_profile)
    st.divider()
    st.subheader(f"日内走势 · 5 分钟{(' · ' + str(result.intraday_session)) if result.intraday_session else ''}")
    if result.intraday.empty:
        st.info(result.intraday_error or "日内分析不可用：尚无可用的常规时段 5 分钟数据。")
    else:
        st.plotly_chart(intraday_chart(result.intraday), use_container_width=True, config=plot_config)
        st.caption("仅常规交易时段 · VWAP 为 5 分钟 OHLCV 估算值 · 浅色 K 线仍在形成。")
        if not result.intraday["vwap"].notna().any():
            st.info("该时段数据不足以计算完整的交易日 VWAP；图中保留实际 K 线。")

with summary_column:
    st.subheader("技术观察")
    st.caption(f"确认日线基准：{result.confirmed_daily_session or '不可用'}")
    for item in result.summaries:
        with st.container(border=True):
            st.markdown(f"**{item.title}**")
            st.write(item.text)
            with st.expander("查看数值与规则"):
                st.write(item.evidence)
                st.caption(f"规则：{item.rule_id}")
    st.caption("日线优先，盘中用于观察当日确认。条件式操作参考见上方「技术强弱」。")

with st.expander("数据明细与口径"):
    st.write("行情时间与读取时间分别记录。日线接口只提供交易日，其收盘时点由交易日历推导；5 分钟时间戳按区间开始处理。")
    st.write("按 America/New_York 和 NYSE 日历处理节假日、夏令时及提前收市。盘前盘后不参与分析；未完成 K 线不参与确认性摘要。")
    st.write("FMP 官方说明日线 close 与日内价格均按拆股调整。实际响应、数据覆盖和当前账户权限决定可用功能。")
    st.markdown("[FMP 数据时区与复权说明](https://site.financialmodelingprep.com/faqs)")
    daily_tab, intraday_tab = st.tabs(["日线数据", "日内数据"])
    with daily_tab:
        st.dataframe(shown_daily.rename(columns={"complete": "已完成", "session": "交易日"}), hide_index=True, use_container_width=True)
    with intraday_tab:
        st.dataframe(result.intraday.rename(columns={"complete": "已完成", "session": "交易日"}), hide_index=True, use_container_width=True)
    ui_ms = (perf_counter() - ui_started) * 1000
    st.session_state["last_ui_render_ms"] = ui_ms
    timings = result.timings
    st.caption(
        f"运行耗时 · 日线 API {timings.daily_api_ms:.1f} ms · 日内 API {timings.intraday_api_ms:.1f} ms · "
        f"行情阶段墙钟 {timings.data_wall_ms:.1f} ms · 校验累计 {timings.validation_ms:.1f} ms · "
        f"分析 {timings.compute_ms:.1f} ms（结构 {timings.support_resistance_ms:.1f} ms） · "
        f"服务端界面生成 {ui_ms:.1f} ms"
    )
    st.caption("API 独立并发，耗时不能直接相加；服务端界面生成包含图表构造与序列化，不含浏览器传输和绘制。")
    st.caption(f"日线缓存：{'命中' if timings.daily_cache_hit else '未命中'} · "
               f"日内缓存：{'命中' if timings.intraday_cache_hit else '未命中'} · "
               f"分析缓存：{'命中' if timings.derived_cache_hit else '未命中'}")

watch_daily_close()
