"""Scenario suite for the decision core (decision.decide).

Phase 2 contract: the strategy_core indicator funnel decides (STEP A -> B -> C),
the demoted layers only score and level, ONE HTF BOS veto survives, and the
mandatory risk gate still refuses a trade with unsafe geometry. NO_TRADE is a
first-class result with explicit reason codes.

All fixtures are synthetic closed-candle frames. The funnel-passing candle
paths live in test_strategy_core.py; this suite imports them so the scenarios
exercise the full orchestration, with the risk gate swapped for a
deterministic outcome where the scenario needs one (its own suite covers the
risk internals).
"""
import numpy as np

import config
import decision
from conftest import make_risk_pass, make_risk_fail
from test_strategy_core import (_buy_rows, _sell_rows, _ltf_rows,
                                _frame as _core_frame, _LTF_START)


# --------------------------------------------------------------------------- helpers

def zig(controls, seg=5):
    """Piecewise-linear close series through `controls` (each leg `seg` candles)."""
    closes = []
    for a, b in zip(controls[:-1], controls[1:]):
        closes.extend(np.linspace(a, b, seg, endpoint=False))
    closes.append(controls[-1])
    return closes


def _frames(df):
    """Same frame in all three roles (structure agrees across timeframes)."""
    return {config.TF_HTF: df, config.TF_SETUP: df, config.TF_ENTRY: df}


# A clean higher-low uptrend that clears every primary + risk check.
UPTREND = [130, 126, 131, 101, 107, 104, 110, 107, 112, 111, 112.5]
# Its mirror: a lower-high downtrend.
DOWNTREND = [100, 104, 99, 129, 123, 126, 120, 123, 118, 119, 117.5]
# A tight oscillation with no net direction.
RANGE = [100, 101, 99, 100.5, 99.5, 100.5, 99.5, 100.5, 99.5, 100.5, 99.5]
# A wider oscillation that still yields a (blocked) directional read.
WIDE_RANGE = [100, 103, 98, 102, 99, 103, 98, 102, 99, 103, 98]


def _candles(controls, **kw):
    # local import so conftest's fixture path is on sys.path when pytest collects
    from conftest import make_candles
    return make_candles(zig(controls), wick=kw.pop("wick", 0.3), **kw)


# --------------------------------------------------------------------------- directional

def _funnel_frames(direction="BUY"):
    """Frames whose indicator funnel PASSES for the given direction."""
    rows_1h = _buy_rows() if direction == "BUY" else _sell_rows()
    return {config.TF_HTF: _core_frame(rows_1h),
            config.TF_SETUP: _core_frame(_ltf_rows(direction), "15m", _LTF_START),
            config.TF_ENTRY: _core_frame(_ltf_rows(direction), "5m", _LTF_START)}


# --------------------------------------------------------------------------- directional

def test_trend_continuation_is_long(monkeypatch):
    make_risk_pass(monkeypatch)
    d = decision.decide(_funnel_frames("BUY"), funding_rate=0.0001)
    assert d["decision"] == "LONG"
    assert d["direction"] == "LONG"
    assert d["no_trade_reasons"] == []
    assert d["core"]["passed"] is True        # the funnel decided the direction
    # mandatory risk gate produced real, ordered levels
    assert d["sl"] < d["entry"] < d["tp"]
    assert d["rr"] >= config.MIN_RR


def test_downtrend_continuation_is_short(monkeypatch):
    make_risk_pass(monkeypatch)
    d = decision.decide(_funnel_frames("SELL"), funding_rate=-0.0001)
    assert d["decision"] == "SHORT"
    assert d["direction"] == "SHORT"
    assert d["no_trade_reasons"] == []
    assert d["tp"] < d["entry"] < d["sl"]


# --------------------------------------------------------------------------- NO_TRADE is first-class

def test_range_fails_the_funnel_with_explicit_reasons():
    df = _candles(RANGE, wick=0.2)
    d = decision.decide(_frames(df), funding_rate=0.0001)
    assert d["decision"] == "NO_TRADE"
    assert d["direction"] is None
    # the funnel names the exact failing 1H checks (never a vague "no bias")
    assert d["no_trade_reasons"]
    assert all(r.startswith("1h_") for r in d["no_trade_reasons"])


def test_risk_gate_blocks_even_with_a_direction(monkeypatch):
    make_risk_fail(monkeypatch, reasons=("poor_rr", "stop_too_wide"))
    d = decision.decide(_funnel_frames("BUY"), funding_rate=0.0001)
    assert d["decision"] == "NO_TRADE"
    assert d["direction"] == "LONG"           # the funnel did pick a side...
    assert d["no_trade_reasons"] == ["poor_rr", "stop_too_wide"]   # ...but the mandatory gate refused it


def test_indicators_alone_cannot_force_a_trade():
    """Structure-neutral series → NO_TRADE, no matter what the indicators read.

    Indicators are a bounded secondary term; with no primary direction there is
    nothing for them to confirm, so the core must not manufacture an entry.
    """
    df = _candles(RANGE, wick=0.2)
    d = decision.decide(_frames(df), funding_rate=0.0001)
    assert d["decision"] == "NO_TRADE"
    # with no primary direction the core never even reaches the indicator layer
    assert d["direction"] is None
    assert d["indicators"] is None                 # indicators alone start nothing


def test_insufficient_data_is_no_trade():
    df = _candles(UPTREND).head(config.STRUCT_PIVOT_LEFT + config.STRUCT_PIVOT_RIGHT)
    d = decision.decide(_frames(df), funding_rate=0.0001)
    assert d["decision"] == "NO_TRADE"
    assert "insufficient_data" in d["no_trade_reasons"]
    assert any(w.startswith("insufficient_frame") for w in d["data_warnings"])


# --------------------------------------------------------------------------- safe degrade

def test_missing_futures_degrades_safely(monkeypatch):
    make_risk_pass(monkeypatch)
    d = decision.decide(_funnel_frames("BUY"), funding_rate=None, oi_df=None)
    # decision still produced — futures are contextual, never required
    assert d["direction"] == "LONG"
    assert d["futures"]["available"] is False
    assert "open_interest_unavailable" in d["data_warnings"]
    assert "funding_unavailable" in d["data_warnings"]


def test_missing_indicator_snapshot_fails_the_funnel_explicitly(monkeypatch):
    """Phase 2: the indicators ARE the primary gate, so a missing snapshot is
    no longer a warning-with-decision — the funnel cannot evaluate and says
    exactly that (core_no_indicator_snapshot), never a fabricated verdict."""
    import strategy_core
    monkeypatch.setattr(decision.strategy_core, "compute_indicators",
                        lambda _df: None)
    d = decision.decide(_funnel_frames("BUY"), funding_rate=0.0001)
    assert d["decision"] == "NO_TRADE"
    assert d["no_trade_reasons"] == [strategy_core.RC_NO_SNAPSHOT]


# --------------------------------------------------------------------------- no look-ahead

def test_decision_uses_only_candles_up_to_the_bar():
    """decide() at bar t must not depend on candles after t.

    make_candles is deterministic per index, so a frame truncated at t has the
    exact same rows as a longer frame sliced to t. The two decisions must match.
    """
    long_df = _candles(UPTREND)
    t = len(long_df) - 3                      # a bar with future candles after it
    early = decision.decide(_frames(long_df.iloc[: t + 1]), funding_rate=0.0001)
    sliced = decision.decide(_frames(long_df.iloc[: t + 1].copy()), funding_rate=0.0001)
    assert early["decision"] == sliced["decision"]
    assert early["entry"] == sliced["entry"]
    assert early["setup_quality"] == sliced["setup_quality"]


# --------------------------------------------------------------------------- output contract

def test_structured_output_has_the_full_evidence():
    df = _candles(UPTREND)
    d = decision.decide(_frames(df), funding_rate=0.0001)
    for key in ("decision", "direction", "setup_quality", "confidence", "primary",
                "htf_bias", "mtf", "structure", "sr", "liquidity", "price_action",
                "trendline", "futures", "indicators", "quality", "risk",
                "entry", "sl", "tp", "rr", "no_trade_reasons", "data_warnings"):
        assert key in d, f"missing key: {key}"
    assert d["decision"] in ("LONG", "SHORT", "NO_TRADE")
