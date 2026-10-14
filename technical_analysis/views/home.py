"""Watchlist home page: every saved ticker's action and score in one table."""
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from data.market_store import freshness_epoch
from data.normalize import session_schedule
from data.watchlist import MAX_SYMBOLS, add_symbol, load_watchlist, remove_symbols, save_watchlist, set_holding
from models import DataError
from services import watchlist_service
from services.watchlist_service import ACTION_ORDER
from ui.client import require_client

NEW_YORK = ZoneInfo("America/New_York")
INTERVALS = {0: "关闭", 300: "每 5 分钟", 900: "每 15 分钟", 1800: "每 30 分钟", 3600: "每 60 分钟"}
BADGES = {"buy": "🟢 买入", "wait": "⚪ 等待", "hold": "🔵 持有观察", "reduce": "🟠 减仓",
          "reduce_sell": "🔴 卖出", "insufficient_data": "⚫ 数据不足", "error": "⚠️ 读取失败"}
CATCH_UP_SECONDS = 300


def _now() -> datetime:
    return datetime.now(NEW_YORK)


def run_batch(symbols: list[str], *, refresh: bool) -> None:
    """Analyze tickers with a progress bar; results merge into this session's rows."""
    st.session_state["watchlist_clock"] = perf_counter()
    if not symbols:
        return
    try:
        client = require_client()
    except DataError as error:
        st.session_state["watchlist_error"] = str(error)
        return
    st.session_state.pop("watchlist_error", None)
    bar = st.progress(0.0, text=f"正在分析 {len(symbols)} 只股票…")
    rows = watchlist_service.analyze_watchlist(
        symbols, client, refresh=refresh,
        progress=lambda done, total, symbol: bar.progress(done / total, text=f"已完成 {symbol}（{done}/{total}）"))
    bar.empty()
    st.session_state.setdefault("watchlist_rows", {}).update(rows)
    st.session_state["watchlist_refreshed_at"] = _now()
    st.session_state["watchlist_clock"] = perf_counter()


def open_detail(symbol: str, holding: bool) -> None:
    st.session_state["open_ticker"] = (symbol, holding)
    st.switch_page("views/detail.py")


def _market_open(now: datetime) -> bool:
    schedule = session_schedule(now.date(), now.date())
    return not schedule.empty and schedule.iloc[0].market_open <= now < schedule.iloc[0].market_close


def _table(items, rows) -> pd.DataFrame:
    records = []
    for item in items:
        row = rows.get(item.symbol)
        if row is None:
            records.append({"代码": item.symbol, "持有": item.holding, "操作参考": "… 待分析", "_order": 9})
            continue
        code, _, reason = row.action(item.holding)
        records.append({
            "代码": item.symbol,
            "持有": item.holding,
            "操作参考": BADGES.get(code, code),
            "综合评分": row.score,
            "入场形态": row.setup or "—",
            "盈亏比": row.reward_risk or "—",
            "评级": row.grade,
            "日线判断": row.daily_label,
            "最新价": row.price,
            "涨跌幅 %": row.change_percent,
            "主要原因": row.error or reason,
            "距吊灯止损 %": None if row.stop_distance is None else row.stop_distance * 100,
            "相对 SPY（3 个月）%": None if row.relative_strength is None else row.relative_strength * 100,
            "下次财报": row.next_earnings,
            "数据交易日": f"{row.session}{' · 盘中暂定' if row.provisional else ''}" if row.session else None,
            "_order": ACTION_ORDER.get(code, 8),
            "_score": -1 if row.score is None else row.score,
        })
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["_score"] = frame.get("_score", pd.Series(-1, index=frame.index)).fillna(-1)
    # Highest score first; tickers without a score (errors, missing data) last.
    return frame.sort_values(["_score", "代码"], ascending=[False, True]).reset_index(drop=True)


st.markdown('<div class="eyebrow">自选股 · 一览</div>', unsafe_allow_html=True)
st.title("今天先看哪几只？")
st.caption("按综合评分排序。每只股票使用与个股分析页相同的规则；操作参考基于已完成日线。点击任意一行打开个股分析。")

items, problem = load_watchlist()
editable = problem is None
if problem:
    st.error(problem)
rows: dict = st.session_state.setdefault("watchlist_rows", {})
for symbol in [symbol for symbol in rows if symbol not in {item.symbol for item in items}]:
    rows.pop(symbol)

with st.container(border=True):
    add_column, refresh_column, auto_column = st.columns([3.2, 1, 1.3], vertical_alignment="bottom")
    with add_column.form("add_ticker", clear_on_submit=True, border=False):
        input_column, button_column = st.columns([3, 1], vertical_alignment="bottom")
        new_symbol = input_column.text_input("添加股票代码", placeholder="例如 TSLA、AMD、BRK.B", max_chars=15)
        added = button_column.form_submit_button("＋ 添加股票", type="primary", use_container_width=True,
                                                 disabled=not editable)
    refresh_all = refresh_column.button("↻ 刷新全部", use_container_width=True, disabled=not items)
    auto_column.selectbox("自动刷新", list(INTERVALS), index=2, key="watchlist_auto_refresh",
                          format_func=INTERVALS.get,
                          help="页面打开时，在美股常规交易时段定时刷新整个自选股；收盘后自动补取当日收盘数据。")

if added and new_symbol.strip():
    try:
        items = add_symbol(items, new_symbol)
        save_watchlist(items)
        st.toast(f"已添加 {items[-1].symbol}。")
        run_batch([items[-1].symbol], refresh=False)
    except DataError as error:
        st.warning(str(error))

if refresh_all:
    run_batch([item.symbol for item in items], refresh=True)
else:
    missing = [item.symbol for item in items if item.symbol not in rows]
    if missing and "watchlist_error" not in st.session_state:
        run_batch(missing, refresh=False)

if st.session_state.get("watchlist_error"):
    st.error(st.session_state["watchlist_error"])

if not items:
    st.info("自选股为空。在上方输入股票代码并点击「添加股票」。")
else:
    actions = [rows[item.symbol].action(item.holding)[0] for item in items if item.symbol in rows]
    counts = st.columns(5)
    for column, (label, codes) in zip(counts, (("买入", {"buy"}), ("等待", {"wait"}), ("持有观察", {"hold"}),
                                              ("减仓 / 卖出", {"reduce", "reduce_sell"}),
                                              ("数据不足 / 失败", {"insufficient_data", "error"}))):
        column.metric(label, sum(code in codes for code in actions))
    table = _table(items, rows)
    shown = table.drop(columns=[column for column in ("_order", "_score") if column in table])
    event = st.dataframe(
        shown, key="watchlist_table", on_select="rerun", selection_mode="single-row",
        hide_index=True, use_container_width=True,
        column_config={
            "持有": st.column_config.CheckboxColumn("持有", help="在下方「编辑自选股」中修改"),
            "综合评分": st.column_config.ProgressColumn("综合评分", min_value=0, max_value=100, format="%.0f",
                                                    help="七项指标加权评分；新开仓需至少 60 分。"),
            "入场形态": st.column_config.TextColumn(help="今日已完成日线出现的入场触发：回踩、站回 SMA20 或放量突破 60 日前高。"),
            "盈亏比": st.column_config.TextColumn(help="上方最近前高的空间 ÷ 到吊灯止损的距离；新开仓需至少 2。"),
            "最新价": st.column_config.NumberColumn(format="%.2f"),
            "涨跌幅 %": st.column_config.NumberColumn(format="%+.2f"),
            "距吊灯止损 %": st.column_config.NumberColumn(format="%.1f", help="收盘价高于吊灯止损（22 日最高价 − 3 ATR）的百分比。"),
            "相对 SPY（3 个月）%": st.column_config.NumberColumn(format="%+.1f", help="近 63 个交易日涨跌幅减去 SPY 同期涨跌幅。"),
            "下次财报": st.column_config.DateColumn(format="YYYY-MM-DD"),
        },
    )
    selected = getattr(getattr(event, "selection", None), "rows", None) or []
    if selected:
        symbol = table.iloc[selected[0]]["代码"]
        open_detail(symbol, next(item.holding for item in items if item.symbol == symbol))
    refreshed = st.session_state.get("watchlist_refreshed_at")
    st.caption(("最近刷新：" + refreshed.strftime("%Y-%m-%d %H:%M:%S %Z") if refreshed else "尚未刷新") +
               "　·　评分是规则排序参考，不是成功概率；在回测验证权重前，请把它当作「先看哪几只」的清单，而不是下单清单。")

    with st.expander(f"编辑自选股（{len(items)}/{MAX_SYMBOLS}）"):
        editor = st.data_editor(
            pd.DataFrame({"代码": [item.symbol for item in items], "我持有": [item.holding for item in items],
                          "移除": [False] * len(items)}),
            key="watchlist_editor", hide_index=True, use_container_width=True, disabled=["代码"] if editable else True,
            column_config={"我持有": st.column_config.CheckboxColumn(help="持有的股票显示持有／减仓／卖出，未持有的显示买入／等待。"),
                           "移除": st.column_config.CheckboxColumn(help="勾选后点击「保存修改」从自选股中移除。")})
        if st.button("保存修改", disabled=not editable):
            updated = list(items)
            for row in editor.itertuples(index=False):
                updated = set_holding(updated, row[0], bool(row[1]))
            updated = remove_symbols(updated, [row[0] for row in editor.itertuples(index=False) if row[2]])
            save_watchlist(updated)
            st.session_state.pop("watchlist_editor", None)
            st.rerun()


@st.fragment(run_every=15)
def watch_watchlist_refresh() -> None:
    """Timed refresh during regular hours, plus one catch-up after each close."""
    current, problem_now = load_watchlist()
    symbols = [item.symbol for item in current]
    interval = st.session_state.get("watchlist_auto_refresh", 900)
    if not symbols or problem_now:
        return
    if not interval:
        st.caption("自动刷新已关闭。可点击「刷新全部」手动更新。")
        return
    now = _now()
    elapsed = perf_counter() - st.session_state.setdefault("watchlist_clock", perf_counter())
    if _market_open(now):
        if elapsed >= interval:
            run_batch(symbols, refresh=True)
            st.rerun()
        st.caption(f"自动刷新已开启 · {INTERVALS[interval]} · 距下次刷新约 {max(1, int(interval - elapsed)) // 60 + 1} 分钟。")
        return
    epoch = freshness_epoch("daily", now)
    stored = st.session_state.get("watchlist_rows", {})
    behind = [symbol for symbol in symbols if symbol in stored and stored[symbol].error is None and
              (stored[symbol].confirmed_session is None or
               (epoch is not None and stored[symbol].confirmed_session < epoch.date()))]
    if behind and elapsed >= CATCH_UP_SECONDS:
        run_batch(symbols, refresh=False)
        st.rerun()
    st.caption("休市期间暂停定时刷新；收盘后会自动补取当日收盘数据。")


watch_watchlist_refresh()
