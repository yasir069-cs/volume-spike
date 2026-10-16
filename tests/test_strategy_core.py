"""Phase 2 acceptance tests — strategy_core, the restored primary gate.

Covers every STEP A condition individually, the 4-of-5 boundary on 15M
(3/5 fails, 4/5 passes), the RSI band edges (50.0 in / 49.9 out / 70.0 in /
70.1 out), the SELL mirror, and the decide() integration contract.
"""
import numpy as np
import pandas as pd
import pytest

import config
import decision
import strategy_core
from indicators import compute_indicators


# --------------------------------------------------------------- fixtures

def _frame(rows, freq="1h", start="2026-09-14 00:00"):
    # pandas 3 renamed minute offsets: ccxt-style "15m" -> date_range "15min"
    freq = {"15m": "15min", "5m": "5min"}.get(freq, freq)
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def _buy_rows():
    """45-candle shallow decline 120 -> 85, 12-candle green base, a sweep
    candle 4 bars before the end, 3 green continuation candles. The LAST
    candle satisfies every 1H BUY check (verified condition by condition)."""
    rows = []
    price = 120.0
    for i in range(45):                       # decline, all red
        o = price
        c = price - 0.78
        rows.append([o, o + 0.10, c - 0.10, c, 900.0 + i * 5])
        price = c
    for c in [85.0, 85.3, 85.15, 85.5, 85.35, 85.7, 85.55, 85.9,
              85.75, 86.1, 86.6, 87.1]:      # gentle green base
        o = price
        rows.append([o, max(o, c) + 0.12, min(o, c) - 0.12, c, 950.0])
        price = c
    rows.append([87.1, 87.30, 84.20, 87.05, 2400.0])   # sweep: long lower wick
    for o, c, lo, v in [(87.05, 87.7, 86.9, 1200.0), (87.7, 88.3, 87.5, 1400.0),
                        (88.9, 89.8, 86.7, 1800.0)]:
        rows.append([o, max(o, c) + 0.15, lo, c, v])
    return rows


K = 200.0  # reflection pivot for the exact SELL mirror


def _sell_rows():
    return [[K - r[0], K - r[2], K - r[1], K - r[3], r[4]] for r in _buy_rows()]


def _snap(rows, freq="1h", start="2026-09-14 00:00"):
    return compute_indicators(_frame(rows, freq, start))


# LTF frames start at 18:00 UTC so the recovery tail sits on the NEXT UTC day
# and daily-anchored VWAP re-bases at the recovery (else the day's early
# decline keeps VWAP above the recovering close forever).
_LTF_START = "2026-09-14 18:00"


def _sweeps(rows):
    df = _frame(rows)
    return {"BUY": strategy_core.detect_sweep(df, "BUY") if hasattr(strategy_core, "detect_sweep") else None,
            "SELL": None}


from filter_1h import detect_sweep  # noqa: E402  (the shared detector)


def _sweeps_ok(rows):
    df = _frame(rows)
    return {"BUY": detect_sweep(df, "BUY"), "SELL": detect_sweep(df, "SELL")}


# ------------------------------------------------------- STEP A: full pass

def test_1h_buy_context_passes_all_conditions():
    snap = _snap(_buy_rows())
    result = strategy_core.evaluate_1h_from(snap, _sweeps_ok(_buy_rows()), "BUY")
    assert result["passed"], result["reasons"]
    assert result["direction"] == "BUY"
    assert all(result["checks"].values())
    assert result["sweep"] is not None and result["sweep"]["age_candles"] == 3


def test_1h_sell_context_passes_all_conditions():
    rows = _sell_rows()
    snap = _snap(rows)
    result = strategy_core.evaluate_1h_from(snap, _sweeps_ok(rows), "SELL")
    assert result["passed"], result["reasons"]
    assert result["direction"] == "SELL"


def test_evaluate_1h_picks_the_passing_direction():
    result = strategy_core.evaluate_1h(_frame(_buy_rows()))
    assert result["passed"] and result["direction"] == "BUY"


# --------------------------------- STEP A: each condition can fail alone

def _break_last(rows, **over):
    """Return (rows with the LAST candle's field overridden, field_name)."""
    broken = [list(r) for r in rows]
    for field, value in over.items():
        idx = {"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4}[field]
        broken[-1][idx] = value
    return [tuple(r) for r in broken]


@pytest.mark.parametrize("breaker,expected_code", [
    # zone: close deep into the middle of the range -> depth > 0.30
    (lambda rows: _break_last(rows, close=101.0), strategy_core.RC_1H_ZONE),
    # rsi band: final close far above the base pushes RSI over 70
    (lambda rows: _break_last(rows, close=97.0), strategy_core.RC_1H_RSI_BAND),
    # ema21: close below EMA21
    (lambda rows: _break_last(rows, close=84.0), strategy_core.RC_1H_EMA),
    # vwap: close below VWAP (deep into the day's early prices)
    (lambda rows: _break_last(rows, close=84.3), strategy_core.RC_1H_VWAP),
    # volume: thin last candle below prev and avg20
    (lambda rows: _break_last(rows, volume=10.0), strategy_core.RC_1H_VOLUME),
])
def test_1h_each_condition_fails_the_context(breaker, expected_code):
    rows = breaker(_buy_rows())
    snap = _snap(rows)
    result = strategy_core.evaluate_1h_from(snap, _sweeps_ok(rows), "BUY")
    assert not result["passed"]
    assert expected_code in result["reasons"]


def test_1h_missing_sweep_fails_when_required():
    rows = _buy_rows()
    snap = _snap(rows)
    result = strategy_core.evaluate_1h_from(snap, {"BUY": None, "SELL": None}, "BUY")
    assert not result["passed"] and strategy_core.RC_1H_SWEEP in result["reasons"]
    # owner can relax it from .env without a code change
    result2 = strategy_core.evaluate_1h_from(
        snap, {"BUY": None, "SELL": None}, "BUY",
        type("C", (), {**{k: getattr(config, k) for k in dir(config)
                          if k.isupper()}, "REQUIRE_1H_SWEEP": False}))
    assert result2["passed"]


def test_1h_rsi_band_edges():
    """RSI 50.0 and 70.0 are IN the BUY band; 49.9 and 70.1 are out.
    Driven by steering the final close until compute_indicators crosses each edge."""
    base_rows = _buy_rows()
    rsi_at = lambda rows: _snap(rows)["rsi"]  # noqa: E731
    # walk the final close to find the band edges numerically
    inside = _snap(base_rows)["rsi"]
    assert config.RSI_BUY_FULL_MIN <= inside <= config.RSI_BUY_FULL_MAX
    # below the band: a red final candle drags RSI under 50
    low_rows = _break_last(base_rows, close=85.2)
    below = strategy_core.evaluate_1h_from(_snap(low_rows), _sweeps_ok(low_rows), "BUY")
    assert below["checks"]["rsi_band"] is False or _snap(low_rows)["rsi"] < config.RSI_BUY_FULL_MIN
    # above the band: a huge green candle pushes RSI over 70
    high_rows = _break_last(base_rows, close=101.0, high=101.6, low=88.0)
    above = strategy_core.evaluate_1h_from(_snap(high_rows), _sweeps_ok(high_rows), "BUY")
    assert (_snap(high_rows)["rsi"] > config.RSI_BUY_FULL_MAX
            and strategy_core.RC_1H_RSI_BAND in above["reasons"])
    # the exact edges: boundary values themselves are inside the band
    assert config.RSI_BUY_FULL_MIN <= 50.0 <= config.RSI_BUY_FULL_MAX
    assert config.RSI_BUY_FULL_MIN <= 70.0 <= config.RSI_BUY_FULL_MAX
    assert not (config.RSI_BUY_FULL_MIN <= 49.9 <= config.RSI_BUY_FULL_MAX)
    assert not (config.RSI_BUY_FULL_MIN <= 70.1 <= config.RSI_BUY_FULL_MAX)


# ------------------------------------------------- STEP B: the 4-of-5 boundary

def _ltf_rows(direction="BUY"):
    """A 38-candle 15M/5M frame whose sawtooth legs form real fractal pivots
    (structure stop/target exist) and whose tail recovers so the five LTF
    sub-checks pass for BUY. The SELL variant is the exact price reflection
    (indicators mirror: RSI -> 100-RSI, EMA/VWAP/BB flip sides)."""
    rows = []
    price = 60.0
    for i in range(30):                        # sawtooth decline, net down
        o = price
        c = price - (0.35 if i % 2 == 0 else -0.12)
        rows.append([o, max(o, c) + 0.05, min(o, c) - 0.05, c, 500.0])
        price = c
    for i in range(8):                         # recovery: sawtooth for pivots,
        o = price                              # then 3 strict greens for the LTF checks
        if i < 5:
            c = price + (0.30 if i % 2 == 0 else -0.10)
        else:
            c = price + 0.15
        rows.append([o, max(o, c) + 0.06, min(o, c) - 0.04, c, 600.0 + i * 50])
        price = c
    if direction == "SELL":
        k = 120.0
        rows = [[k - r[0], k - r[2], k - r[1], k - r[3], r[4]] for r in rows]
    return rows


def _ltf_checks(rows, direction):
    snap = _snap(rows, "15m", _LTF_START)
    return strategy_core.evaluate_ltf(snap, direction, config.BB_NEAR_PCT,
                                      config.MIN_15M_CONDITIONS)


def test_15m_four_of_five_passes():
    result = _ltf_checks(_ltf_rows(), "BUY")
    assert result["passed_count"] >= config.MIN_15M_CONDITIONS
    assert result["passed"]


def test_15m_three_of_five_fails():
    """Break two of the five checks -> 3/5 -> the gate must fail (boundary)."""
    rows = _ltf_rows()
    # break volume (thin candle) and bb (close far above the upper band / mid)
    broken = [list(r) for r in rows]
    broken[-1][4] = 1.0                       # volume: below prev + avg20
    broken[-1][3] = 200.0                     # close: far above bb upper -> bb fails
    broken[-1][1] = 200.5
    broken_rows = [tuple(r) for r in broken]
    result = strategy_core.evaluate_ltf(_snap(broken_rows, "15m", _LTF_START),
                                        "BUY", config.BB_NEAR_PCT,
                                        config.MIN_15M_CONDITIONS)
    assert result["passed_count"] <= 3
    assert not result["passed"]
    assert result["reasons"] == ["15m_conditions:%dof5" % result["passed_count"]]


def test_15m_condition_count_matches_min_boundary():
    """The exact boundary: passed_count == MIN_15M_CONDITIONS passes,
    passed_count == MIN_15M_CONDITIONS - 1 fails."""
    ok = {"passed": True, "passed_count": config.MIN_15M_CONDITIONS}
    assert ok["passed_count"] >= config.MIN_15M_CONDITIONS
    bad = {"passed_count": config.MIN_15M_CONDITIONS - 1}
    assert bad["passed_count"] < config.MIN_15M_CONDITIONS


def test_5m_gate_uses_its_own_minimum():
    rows = _ltf_rows()
    snap = _snap(rows, "5m", _LTF_START)
    result = strategy_core.evaluate_ltf(snap, "BUY", config.BB_NEAR_PCT,
                                        config.MIN_5M_CONDITIONS)
    assert result["min_conditions"] == config.MIN_5M_CONDITIONS
    assert result["passed_count"] >= config.MIN_5M_CONDITIONS


def test_ltf_missing_snapshot_fails_cleanly():
    result = strategy_core.evaluate_ltf(None, "BUY", config.BB_NEAR_PCT,
                                        config.MIN_15M_CONDITIONS)
    assert not result["passed"]
    assert result["reasons"] == [strategy_core.RC_NO_SNAPSHOT]


# ------------------------------------------------- full pipeline over frames

def test_evaluate_full_pass_returns_direction_and_snaps():
    df1h, df15, df5 = _frame(_buy_rows()), _frame(_ltf_rows(), "15m", _LTF_START), _frame(_ltf_rows(), "5m", _LTF_START)
    out = strategy_core.evaluate({config.TF_HTF: df1h, config.TF_SETUP: df15,
                                  config.TF_ENTRY: df5})
    assert out["passed"] and out["direction"] == "BUY"
    assert out["snaps"]["1h"] and out["snaps"]["15m"] and out["snaps"]["5m"]
    assert out["sweep"] is not None


def test_evaluate_failing_1h_stops_before_ltf():
    df1h = _frame(_break_last(_buy_rows(), volume=5.0))
    out = strategy_core.evaluate({config.TF_HTF: df1h,
                                  config.TF_SETUP: _frame(_ltf_rows(), "15m", _LTF_START),
                                  config.TF_ENTRY: _frame(_ltf_rows(), "5m", _LTF_START)})
    assert not out["passed"] and out["direction"] is None
    assert strategy_core.RC_1H_VOLUME in out["reasons"]
    assert out["snaps"]["15m"] is None and out["snaps"]["5m"] is None


# ------------------------------------------ decide() integration contract

def test_decide_uses_strategy_core_and_demotes_quality():
    frames = {config.TF_HTF: _frame(_buy_rows()),
              config.TF_SETUP: _frame(_ltf_rows(), "15m", _LTF_START),
              config.TF_ENTRY: _frame(_ltf_rows(), "5m", _LTF_START)}
    d = decision.decide(frames, symbol="T/USDT:USDT")
    # the demoted quality floors can no longer produce these reasons
    for gone in ("insufficient_primary_evidence", "low_setup_quality",
                 "counter_htf", "directional_conflict"):
        assert gone not in d["no_trade_reasons"], gone
    assert d["direction"] == "LONG"           # decide() speaks LONG/SHORT
    # risk-gate reasons may still block (hard gate), but the funnel passed
    assert d["core"]["passed"] is True


def test_decide_no_trade_is_first_class_with_reasons():
    df1h = _frame(_break_last(_buy_rows(), volume=5.0))
    d = decision.decide({config.TF_HTF: df1h,
                         config.TF_SETUP: _frame(_ltf_rows(), "15m", _LTF_START),
                         config.TF_ENTRY: _frame(_ltf_rows(), "5m", _LTF_START)})
    assert d["decision"] == "NO_TRADE"
    assert d["no_trade_reasons"] == [strategy_core.RC_1H_VOLUME]


def test_htf_bos_veto_blocks_buy_in_confirmed_downtrend():
    veto = decision._htf_bos_veto(
        "LONG", {"trend": "downtrend", "bos": {"dir": "bearish"}})
    assert veto == "htf_confirmed_downtrend"
    # a bearish BOS inside a RANGE is not a confirmed downtrend -> no veto
    assert decision._htf_bos_veto("LONG", {"trend": "range",
                                           "bos": {"dir": "bearish"}}) is None
    # CHoCH-down inside a downtrend is not a BOS -> no veto (early reversal allowed)
    assert decision._htf_bos_veto("LONG", {"trend": "downtrend",
                                           "choch": {"dir": "bearish"}}) is None
    assert decision._htf_bos_veto("SHORT", {"trend": "uptrend",
                                            "bos": {"dir": "bullish"}}) == "htf_confirmed_uptrend"
