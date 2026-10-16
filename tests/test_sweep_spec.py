"""Phase 3 acceptance tests — the liquidation sweep EXACT 4-condition spec.

The spec (owner's words): a sweep candle must satisfy ALL FOUR of
  1. the wick pierces the last-20 swing low (bullish) / swing high (bearish)
  2. the body closes back inside (above the swing low / below the swing high)
  3. the wick is > 2x the body            (config.SWEEP_WICK_BODY_RATIO = 2.0)
  4. the volume is > 1.5x the 20-candle average   (config.SWEEP_VOL_RATIO = 1.5)

Plus the Phase 3 gate: a sweep older than SWEEP_MAX_AGE_CANDLES (5) is not
actionable — STEP A treats it as no sweep and the setup HOLDs.

Structure: 4 passing sets + 4 single-condition-fail rejections + the age gate
+ the strict-inequality boundaries. Every candle below is hand-built so each
condition is controlled in isolation.
"""
import pandas as pd
import pytest

import config
import strategy_core
from filter_1h import detect_sweep


def _frame(rows, freq="1h"):
    freq = {"15m": "15min", "5m": "5min"}.get(freq, freq)
    idx = pd.date_range("2026-09-14 00:00", periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


BASE_LOW = 99.90     # the flat base's swing low (pierced by a bullish sweep)
BASE_HIGH = 100.10   # the flat base's swing high (pierced by a bearish sweep)


def _base(n=24):
    """Flat base: open=close=100, tiny wicks, volume 1000 — never sweeps by
    itself (equal lows/highs never STRICTLY pierce their own window)."""
    return [[100.0, 100.05, BASE_LOW, 100.0, 1000.0] for _ in range(n)]


def _with_last(base, candle):
    rows = [list(r) for r in base]
    rows.append(candle)
    return _frame([tuple(r) for r in rows])


# ------------------------------------------------------- the 4 passing sets

def test_pass_1_textbook_bullish_sweep():
    """Deep lower wick pierces the base swing low, body closes back above it,
    wick 21x body, volume 1.6x avg20 — all four hold."""
    df = _with_last(_base(), [100.20, 100.40, 98.00, 100.10, 1600.0])
    s = detect_sweep(df, "BUY")
    assert s is not None
    assert s["age_candles"] == 0
    assert s["wick_body_ratio"] > config.SWEEP_WICK_BODY_RATIO
    assert s["volume_ratio"] > config.SWEEP_VOL_RATIO


def test_pass_2_textbook_bearish_sweep():
    """Mirror: upper wick pierces the base swing high, body closes back below,
    wick 21x body, volume 1.6x avg20."""
    df = _with_last(_base(), [99.80, 102.00, 99.60, 99.90, 1600.0])
    s = detect_sweep(df, "SELL")
    assert s is not None
    assert s["age_candles"] == 0
    assert s["wick_body_ratio"] > config.SWEEP_WICK_BODY_RATIO
    assert s["volume_ratio"] > config.SWEEP_VOL_RATIO


def test_pass_3_bullish_sweep_at_exact_max_age():
    """A valid sweep exactly SWEEP_MAX_AGE_CANDLES (5) old is still actionable."""
    rows = _base()
    rows.append([100.20, 100.40, 98.00, 100.10, 1600.0])       # the sweep
    for _ in range(5):                                          # 5 candles after
        rows.append([100.0, 100.05, 99.95, 100.02, 1000.0])
    df = _frame([tuple(r) for r in rows])
    s = detect_sweep(df, "BUY")
    assert s is not None and s["age_candles"] == 5
    result = strategy_core._sweep_ok(s, config)
    assert result is True


def test_pass_4_bearish_sweep_with_bigger_body_still_qualifies():
    """Body 0.5, wick 2.1 -> ratio 4.2x (above the 2x bar), volume 1.6x —
    a larger-bodied sweep passes when the wick dominance holds."""
    df = _with_last(_base(), [99.40, 102.00, 99.30, 99.90, 1600.0])
    s = detect_sweep(df, "SELL")
    assert s is not None
    # wick = high - max(open, close) = 102.0 - 99.9 = 2.1; body = 0.5 -> 4.2x
    assert s["wick_body_ratio"] > config.SWEEP_WICK_BODY_RATIO
    assert s["volume_ratio"] > config.SWEEP_VOL_RATIO


# --------------------------------- the 4 single-condition-fail rejections

def test_fail_1_wick_not_dominant_enough():
    """Condition 3 fails alone: wick/body 1.8x (< 2x) while pierce, close-back
    and volume all hold -> rejected."""
    # body = 101.5 - 100.5 = 1.0; wick = 100.5 - 98.7 = 1.8 -> 1.8x
    df = _with_last(_base(), [101.50, 101.60, 98.70, 100.50, 1600.0])
    assert detect_sweep(df, "BUY") is None


def test_fail_2_volume_below_1_5x_average():
    """Condition 4 fails alone: 1.4x average volume while the other three
    hold -> rejected."""
    # same geometry as pass_1 (wick 21x body) but volume 1400 = 1.4x avg20
    df = _with_last(_base(), [100.20, 100.40, 98.00, 100.10, 1400.0])
    assert detect_sweep(df, "BUY") is None


def test_fail_3_body_does_not_close_back_inside():
    """Condition 2 fails alone: closes BELOW the swept swing low (a breakdown,
    not a sweep) while pierce, wick dominance (3x) and volume hold -> rejected."""
    # o=99.0 c=98.5: body 0.5, wick = 98.5-97.0 = 1.5 -> 3x; low pierces 99.9
    df = _with_last(_base(), [99.00, 99.05, 97.00, 98.50, 1600.0])
    assert detect_sweep(df, "BUY") is None


def test_fail_4_wick_does_not_pierce_the_swing():
    """Condition 1 fails alone: the low stays ABOVE the base swing low while
    close-back, wick dominance (2.6x) and volume hold -> rejected."""
    # o=101.4 c=101.0: body 0.4, wick = 101.0-99.95 = 1.05 -> 2.6x; low 99.95 > 99.9
    df = _with_last(_base(), [101.40, 101.60, 99.95, 101.00, 1600.0])
    assert detect_sweep(df, "BUY") is None


# ----------------------------------------------------- strict-inequality edges

def test_boundaries_are_strict_inequalities():
    """Exactly 2x wick/body and exactly 1.5x volume both FAIL — the spec says
    '>', not '>='."""
    # wick exactly 2x body: body 1.0, wick 2.0 (low = 98.0, close 100.0)
    df = _with_last(_base(), [101.00, 101.10, 98.00, 100.00, 1600.0])
    assert detect_sweep(df, "BUY") is None
    # volume exactly 1.5x: pass_1 geometry with volume 1500
    df2 = _with_last(_base(), [100.20, 100.40, 98.00, 100.10, 1500.0])
    assert detect_sweep(df2, "BUY") is None


# --------------------------------------------------------- the age gate

def test_sweep_older_than_max_age_fails_the_context_gate():
    """A valid sweep 6 candles old is stale: STEP A treats it as no sweep and
    the 1H context fails with 1h_sweep — HOLD as a real gate."""
    rows = _base()
    rows.append([100.20, 100.40, 98.00, 100.10, 1600.0])       # the sweep
    for _ in range(6):                                          # 6 candles after
        rows.append([100.0, 100.05, 99.95, 100.02, 1000.0])
    df = _frame([tuple(r) for r in rows])
    s = detect_sweep(df, "BUY")
    assert s is not None and s["age_candles"] == 6
    assert strategy_core._sweep_ok(s, config) is False
    # and through the full STEP A check:
    from indicators import compute_indicators
    snap = compute_indicators(df)
    result = strategy_core.evaluate_1h_from(snap, {"BUY": s, "SELL": None}, "BUY")
    assert not result["passed"]
    assert strategy_core.RC_1H_SWEEP in result["reasons"]


def test_age_gate_is_env_tunable():
    """SWEEP_MAX_AGE_CANDLES is a named .env-overridable constant: raising it
    to 6 makes the same 6-candle-old sweep actionable again."""
    rows = _base()
    rows.append([100.20, 100.40, 98.00, 100.10, 1600.0])
    for _ in range(6):
        rows.append([100.0, 100.05, 99.95, 100.02, 1000.0])
    df = _frame([tuple(r) for r in rows])
    s = detect_sweep(df, "BUY")
    cfg6 = type("C", (), {**{k: getattr(config, k) for k in dir(config) if k.isupper()},
                          "SWEEP_MAX_AGE_CANDLES": 6})
    assert strategy_core._sweep_ok(s, cfg6) is True


def test_missing_sweep_still_fails_when_required():
    """REQUIRE_1H_SWEEP stays the master switch: no sweep at all -> gate fails
    (the NO_SWEEP_CONFIDENCE_CAP remains display/tier-only downstream)."""
    assert strategy_core._sweep_ok(None, config) is False
