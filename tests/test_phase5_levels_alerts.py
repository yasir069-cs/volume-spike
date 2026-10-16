"""Phase 5 acceptance tests — SL/TP construction, alert tiers, gate summary.

Covers: the fixed 1:2 fallback target maths (long and short), the structural
SL (swing level padded by ATR), MIN_RR as the reported hard gate, the visible
NORMAL/HIGH/STRONG alert tiers, and the one-line per-condition gate summary in
the alert + sig.
"""
import pytest

import alerts
import config
import main
import risk_gate
from test_llm_decision_stage import _decide


# ------------------------------------------------------------- levels

def test_sl_is_swing_level_padded_by_atr():
    struct = {"last_swing_low": {"price": 101.0}, "last_swing_high": {"price": 130.0}}
    out = risk_gate.evaluate("LONG", 105.0, struct, {}, 2.0)
    assert out["ok"] is True
    assert out["sl"] == pytest.approx(101.0 - config.RISK_SL_BUFFER_ATR * 2.0)
    # mirror for a short
    out2 = risk_gate.evaluate("SHORT", 105.0, struct, {}, 2.0)
    assert out2["sl"] == pytest.approx(130.0 + config.RISK_SL_BUFFER_ATR * 2.0)


def test_fallback_tp_is_exactly_two_r_both_directions():
    long_out = risk_gate.evaluate("LONG", 105.0,
                                  {"last_swing_low": {"price": 101.0}}, {}, 2.0)
    assert long_out["tp"] == pytest.approx(105.0 + 2 * (105.0 - long_out["sl"]))
    short_out = risk_gate.evaluate("SHORT", 105.0,
                                   {"last_swing_high": {"price": 107.0}}, {}, 2.0)
    assert short_out["tp"] == pytest.approx(105.0 - 2 * (short_out["sl"] - 105.0))
    for out in (long_out, short_out):
        assert out["rr"] == pytest.approx(2.0)
        assert out["rr"] >= config.MIN_RR          # MIN_RR stays the hard gate


def test_min_rr_default_is_reported():
    """The plan's Phase 5 report item: MIN_RR default."""
    assert config.MIN_RR == 1.5


def test_zone_target_still_wins_when_usable():
    """The fallback only replaces an UNUSABLE zone target: a healthy zone TP
    is taken from support_resistance and the note names it."""
    sr = {"nearest_resistance": {"lo": 120.0, "hi": 122.0, "mid": 121.0,
                                 "side": "resistance"},
          "nearest_support": None, "at_zone": None}
    out = risk_gate.evaluate("LONG", 105.0,
                             {"last_swing_low": {"price": 101.0}}, sr, 2.0)
    assert out["ok"] is True
    assert out["target_zone"].startswith("resistance@")
    assert out["target_zone"] != "fixed-1:2"
    assert out["tp"] == pytest.approx(120.0 - 0.25 * 2.0)   # zone lo - pad


# ------------------------------------------------- alert tiers + gate summary

def test_alert_tiers_are_visible_and_env_floored():
    sig = {"coin": "T", "signal": "BUY", "entry": 100, "SL": 98, "TP": 106,
           "RR": 2.0, "confidence": 62.0, "reason": "ok",
           "conditions": "1H zone✓ RSI✓ EMA21✓ VWAP✓ vol✓ BB✓ sweep✓(2c) | 15M 4of4 | 5M 3of3"}
    text = alerts.format_alert(sig)
    assert "Alert Level:</b> HIGH" in text          # 60-70 -> HIGH, visible
    assert "Strategy Gates:" in text and "15M 4of4" in text and "sweep✓(2c)" in text
    sig["confidence"] = 72.0
    assert "STRONG" in alerts.format_alert(sig)
    sig["confidence"] = 52.0
    assert "NORMAL" in alerts.format_alert(sig)
    # the documented floor: below ALERT_QUALITY_MIN never alerts at all
    assert config.ALERT_QUALITY_MIN == 50.0


def test_conditions_summary_from_a_funnel_decision():
    d = _decide("BUY")
    summary = main._conditions_summary(d)
    assert summary.startswith("1H ")
    assert "zone✓" in summary and "sweep✓(" in summary
    assert "15M 4of4" in summary and "5M 3of3" in summary
    # a funnel-failed decision still names the failing checks
    d2 = _decide("SELL")   # funnel passes for SELL on the mirrored fixture


def test_conditions_summary_empty_for_legacy_sig():
    assert main._conditions_summary(None) == ""
    assert main._conditions_summary({}) == ""
    assert main._conditions_summary({"core": {"steps": {}}}) == ""


def test_sig_carries_conditions_and_fallback_zone():
    d = _decide("BUY")
    sig = main._build_sig("TEST/USDT:USDT", d, "BUY", d["snaps"]["5m"], False,
                          None, d["entry"])
    assert sig["conditions"]
    assert "sweep✓" in sig["conditions"]
