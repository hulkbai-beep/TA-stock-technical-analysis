"""End-to-end support/resistance and chart checks using isolated fixtures."""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from analysis.support_resistance.engine import analyze_support_resistance
from analysis.support_resistance.models import SRConfig
from data.normalize import session_schedule
from models import DataError, RawDataset
from services.analysis_service import analyze, support_resistance_for
from ui.charts import daily_chart, volume_profile_chart

NY = ZoneInfo("America/New_York")


@pytest.fixture
def market_frame():
    schedule = session_schedule("2025-01-02", "2026-10-12")
    close = 100 + 8 * np.cos(np.arange(len(schedule)) * np.pi / 10)
    return pd.DataFrame({
        "session": list(schedule.index), "timestamp": list(schedule["market_close"]),
        "bar_end": list(schedule["market_close"]), "complete": True,
        "open": close - 0.2, "high": close + 1, "low": close - 1, "close": close,
        "volume": 1000.0, "sma20": 100.0, "sma50": 100.0, "rsi14": 50.0,
    })


def test_chart_zone_boundaries_and_avwap_values_equal_backend_outputs(market_frame):
    result = analyze_support_resistance(market_frame, current_price=104)
    assert result.supports and result.resistances and result.anchors
    shown = market_frame.tail(126)
    figure = daily_chart(shown, result, show_profile=False)
    zones = (*result.supports, *result.resistances, *result.interactions)
    assert {(shape.y0, shape.y1) for shape in figure.layout.shapes} == {(zone.lower, zone.upper) for zone in zones}
    assert all(shape.yref == "y" for shape in figure.layout.shapes)
    for anchor in result.anchors:
        expected = [point for point in anchor.points if point.session in set(shown["session"])]
        trace = next(trace for trace in figure.data if isinstance(trace.meta, dict) and trace.meta.get("anchor_id") == anchor.anchor_id)
        assert list(trace.x) == [str(point.session) for point in expected]
        assert list(trace.y) == pytest.approx([point.value for point in expected])
        assert all(point.session >= anchor.confirmed_session for point in expected)
    hidden = daily_chart(shown, result, show_zones=False, show_profile=False, show_avwap=False)
    assert not hidden.layout.shapes
    assert not any(isinstance(trace.meta, dict) and trace.meta.get("anchor_id") == anchor.anchor_id
                   for trace in hidden.data for anchor in result.anchors)
    profile_only = daily_chart(shown, result, show_zones=False, show_profile=True, show_avwap=False)
    assert {(shape.y0, shape.y1) for shape in profile_only.layout.shapes} == {
        (value, value) for value in (result.profile.poc, result.profile.val, result.profile.vah)}
    profile_figure = volume_profile_chart(result.profile)
    assert sum(profile_figure.data[0].x) == pytest.approx(result.profile.total_volume)


def test_service_computes_default_and_controls_reuse_existing_requests(market_frame):
    now = datetime(2026, 10, 12, 17, tzinfo=NY)

    class CountingClient:
        def __init__(self):
            self.daily_calls = self.intraday_calls = 0

        def fetch_daily(self, symbol, now=None, refresh=False):
            self.daily_calls += 1
            records = [{"date": str(row.session), **{column: getattr(row, column)
                       for column in ("open", "high", "low", "close", "volume")}}
                       for row in market_frame.itertuples()]
            return RawDataset(records, now, "test-daily")

        def fetch_intraday(self, symbol, now=None, refresh=False):
            self.intraday_calls += 1
            raise DataError("测试权限不足。", "permission_denied")

    client = CountingClient()
    result = analyze("TEST", client, now=now)
    assert result.support_resistance.available
    assert result.support_resistance.actual_sessions == 252
    assert support_resistance_for(result) is result.support_resistance
    before = result.daily.copy(deep=True)
    six_month = support_resistance_for(result, SRConfig(lookback_sessions=126, max_zones_per_side=2))
    assert six_month.actual_sessions == 126
    assert len(six_month.supports) <= 2 and len(six_month.resistances) <= 2
    assert (client.daily_calls, client.intraday_calls) == (1, 1)
    pd.testing.assert_frame_equal(before, result.daily)


def test_provisional_price_only_changes_position_not_daily_validity(market_frame):
    first = analyze_support_resistance(market_frame, current_price=100)
    provisional = analyze_support_resistance(market_frame, current_price=120, reference_provisional=True)
    assert provisional.reference_provisional
    assert {zone.zone_id: (zone.status, zone.confirmed_role, zone.breakout_session, zone.retest_session)
            for zone in first.zones} == {
                zone.zone_id: (zone.status, zone.confirmed_role, zone.breakout_session, zone.retest_session)
                for zone in provisional.zones}
    assert any(a.kind != b.kind for a, b in zip(first.zones, provisional.zones))
