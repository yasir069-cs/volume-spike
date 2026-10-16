"""Strategy core — the owner's indicator funnel, restored as the PRIMARY gate.

Phase 2 of the alignment plan. Until now the bot decided on market structure
alone (swing bias -> direction) and the handwritten note's indicators were a
bounded afterthought; the result was a 100% NO_TRADE funnel (0 alerts in the
owner's live scan, 0 signals over 999 backtested BTC bars). This module flips
the hierarchy back to what the owner actually wrote:

    STEP A — 1H context PASS/FAIL  (decides the direction; fail -> skip the
             symbol before any 15M/5M fetch is paid for)
    STEP B — 15M confirmation      (need MIN_15M_CONDITIONS of 5)
    STEP C — 5M entry conditions   (need MIN_5M_CONDITIONS of 5)

STEP A conditions (BUY; SELL mirrors every one of them):
    zone      price in the bottom CHECK_1H_ZONE_MAX_PCT (0.30) of the last-50
              candle range            (scoring.ZONE_FULL_PCT, the note's "Bottom")
    rsi       RSI(14) inside [RSI_BUY_FULL_MIN, RSI_BUY_FULL_MAX] = 50-70 AND
              rising (SELL: inside 35-50 AND falling)
    ema21     close above EMA21       (SELL: below)
    vwap      close above daily VWAP  (SELL: below)
    volume    last closed candle's volume above the previous candle's
              (SELL identical — "volume increase" is direction-agnostic)
    bb        near/touching the lower Bollinger band within BB_NEAR_PCT_1H, or
              closing in the lower half of the bands (SELL: upper band / upper half)
    sweep     a liquidation sweep aligned with the direction exists within
              SWEEP_SEARCH_CANDLES (filter_1h.detect_sweep), when
              REQUIRE_1H_SWEEP is on (default on; Phase 3 re-specs the sweep
              itself — the gate here only demands that one EXISTS)

STEP B / STEP C share the same five sub-checks on their timeframe
(RSI-trend, EMA21 side, VWAP side, volume, Bollinger side) counted as
X-of-5 against MIN_15M_CONDITIONS (4) and MIN_5M_CONDITIONS (3).
Every number is a named config constant; several reuse the constants the
secondary layer already had so there is exactly one authority per threshold.

Every check returns an explicit reason code on failure — NO_TRADE stays a
first-class result and the funnel histogram stays readable (scripts/
why_no_signals.py maps these codes to the "strategy core" layer).

Demotion contract (Phase 2): market_structure / support_resistance /
liquidity / trendlines / mtf / direction_gate no longer veto. They keep two
jobs — evidence for setup_quality SCORING, and LEVELS for the risk gate
(structure SL, opposing-zone TP) — plus exactly ONE veto that survived the
collapse of the three direction systems into this module:

    single HTF veto — block a BUY only when the 1H structure shows a
    CONFIRMED downtrend (trend == "downtrend") WITH a bearish BOS; mirror for
    a SELL. Implemented in decision.decide() (it owns the structure reads);
    documented there.

Purity: everything here reads closed-candle frames / snapshots already in
memory. No market access, no look-ahead — the same evaluate() runs in live
and in backtest replay.
"""
from typing import Optional

import pandas as pd

import config
from filter_1h import detect_sweep
from indicators import compute_indicators

# Direction evaluated for STEP A; the SELL path is the exact mirror.
_DIRECTIONS = ("BUY", "SELL")

# Reason codes (stable strings — they land in signals_log.no_trade_reason)
RC_NO_SNAPSHOT = "core_no_indicator_snapshot"
RC_1H_ZONE = "1h_zone"
RC_1H_RSI_BAND = "1h_rsi_band"
RC_1H_RSI_TREND = "1h_rsi_trend"
RC_1H_EMA = "1h_ema21"
RC_1H_VWAP = "1h_vwap"
RC_1H_VOLUME = "1h_volume"
RC_1H_BB = "1h_bb"
RC_1H_SWEEP = "1h_sweep"
RC_15M_CONDITIONS = "15m_conditions"
RC_5M_CONDITIONS = "5m_conditions"


def _zone_ok(snap: dict, direction: str) -> bool:
    """The note's "Bottom to inbetween" starting half: strictly the favourable
    30% of the 50-candle range (ZONE_FULL_PCT). SELL mirrors from the top."""
    depth = snap["range_pos"] if direction == "BUY" else 1.0 - snap["range_pos"]
    return depth <= config.ZONE_FULL_PCT


def _rsi_ok(snap: dict, direction: str) -> bool:
    """RSI inside the note's band AND moving in the trade's direction.
    BUY 50-70 rising, SELL 35-50 falling (RSI_BUY_FULL_* / RSI_SELL_FULL_*)."""
    if direction == "BUY":
        in_band = config.RSI_BUY_FULL_MIN <= snap["rsi"] <= config.RSI_BUY_FULL_MAX
        trending = snap["rsi"] > snap["rsi_prev"]
    else:
        in_band = config.RSI_SELL_FULL_MIN <= snap["rsi"] <= config.RSI_SELL_FULL_MAX
        trending = snap["rsi"] < snap["rsi_prev"]
    return in_band and trending


def _ema_ok(snap: dict, direction: str) -> bool:
    return snap["close"] > snap["ema21"] if direction == "BUY" \
        else snap["close"] < snap["ema21"]


def _vwap_ok(snap: dict, direction: str) -> bool:
    return snap["close"] > snap["vwap"] if direction == "BUY" \
        else snap["close"] < snap["vwap"]


def _volume_ok(snap: dict, direction: str) -> bool:
    """The note's "volume increase": above the previous candle. Sustained
    above the 20-average also counts (a strong candle on a flat tape)."""
    if snap["volume"] > snap["volume_prev"]:
        return True
    return snap["volume_avg20"] > 0 and snap["volume"] > snap["volume_avg20"]


def _bb_ok(snap: dict, direction: str, near_pct: float) -> bool:
    """Near/touching the favourable band, or already back inside the favourable
    half. Uses the same touching rule the graded scorer uses (scoring.bb_score):
    the candle's range must overlap the band neighbourhood."""
    if direction == "BUY":
        band = snap["bb_lower"]
        near = snap["low"] <= band * (1 + near_pct) and snap["high"] >= band * (1 - near_pct)
        return near or snap["close"] <= snap["bb_mid"]
    band = snap["bb_upper"]
    near = snap["high"] >= band * (1 - near_pct) and snap["low"] <= band * (1 + near_pct)
    return near or snap["close"] >= snap["bb_mid"]


# ----------------------------------------------------------------- STEP A

def evaluate_1h(df_or_snap, direction: Optional[str] = None, cfg=config) -> dict:
    """STEP A — the 1H context pass/fail, per direction.

    Accepts a closed-candle 1H frame (computes the snapshot + sweep itself) or
    a precomputed (snap, sweep) pair via :func:`evaluate_1h_from`. With
    `direction=None` both directions are tried and the first that passes all
    required checks wins (the conditions are mutually exclusive by
    construction — the RSI bands and the range halves cannot both hold).
    """
    if isinstance(df_or_snap, pd.DataFrame):
        snap = compute_indicators(df_or_snap)
        if snap is None:
            return {"passed": False, "direction": None, "checks": {},
                    "reasons": [RC_NO_SNAPSHOT], "sweep": None, "indicators": None}
        sweep_buy = detect_sweep(df_or_snap, "BUY")
        sweep_sell = detect_sweep(df_or_snap, "SELL")
    else:  # (snap, sweep_by_direction) precomputed
        snap, sweeps = df_or_snap
        sweep_buy, sweep_sell = sweeps.get("BUY"), sweeps.get("SELL")

    order = (direction,) if direction in _DIRECTIONS else _DIRECTIONS
    best: Optional[dict] = None
    for d in order:
        result = evaluate_1h_from(snap, {"BUY": sweep_buy, "SELL": sweep_sell}, d, cfg)
        if result["passed"]:
            return result
        if best is None or len(result["reasons"]) < len(best["reasons"]):
            best = result
    return best  # closest direction, with its explicit failing checks


def _sweep_ok(sweep: Optional[dict], cfg) -> bool:
    """The STEP A sweep gate (Phase 3).

    With REQUIRE_1H_SWEEP on, a valid aligned sweep must exist AND be recent:
    a sweep older than SWEEP_MAX_AGE_CANDLES (default 5) is stale liquidity
    news — the setup HOLDs (this is a real gate, not a display note). The
    NO_SWEEP_CONFIDENCE_CAP stays for display/tier purposes but can no longer
    rescue a missing-or-stale sweep, because the 1H context itself fails."""
    if not cfg.REQUIRE_1H_SWEEP:
        return True
    if sweep is None:
        return False
    return sweep.get("age_candles", 0) <= cfg.SWEEP_MAX_AGE_CANDLES


def evaluate_1h_from(snap: dict, sweeps: dict, direction: str, cfg=config) -> dict:
    """STEP A for one direction over a precomputed snapshot + sweep pair."""
    sweep = sweeps.get(direction)
    checks = {
        "zone": _zone_ok(snap, direction),
        "rsi_band": (config.RSI_BUY_FULL_MIN <= snap["rsi"] <= config.RSI_BUY_FULL_MAX)
                    if direction == "BUY"
                    else (config.RSI_SELL_FULL_MIN <= snap["rsi"] <= config.RSI_SELL_FULL_MAX),
        "rsi_trend": snap["rsi"] > snap["rsi_prev"] if direction == "BUY"
                     else snap["rsi"] < snap["rsi_prev"],
        "ema21": _ema_ok(snap, direction),
        "vwap": _vwap_ok(snap, direction),
        "volume": _volume_ok(snap, direction),
        "bb": _bb_ok(snap, direction, config.BB_NEAR_PCT_1H),
        "sweep": _sweep_ok(sweep, cfg),
    }
    code_map = {
        "zone": RC_1H_ZONE, "rsi_band": RC_1H_RSI_BAND, "rsi_trend": RC_1H_RSI_TREND,
        "ema21": RC_1H_EMA, "vwap": RC_1H_VWAP, "volume": RC_1H_VOLUME,
        "bb": RC_1H_BB, "sweep": RC_1H_SWEEP,
    }
    reasons = [code_map[name] for name, ok in checks.items() if not ok]
    return {
        "passed": not reasons,
        "direction": direction if not reasons else None,
        "checks": checks,
        "reasons": reasons,
        "sweep": sweep,
        "indicators": snap,
    }


# ----------------------------------------------------------- STEP B / C

_LTF_CHECKS = ("rsi_trend", "ema21", "vwap", "volume", "bb")


def evaluate_ltf(snap: Optional[dict], direction: str, near_pct: float,
                 min_conditions: int, cfg=config) -> dict:
    """STEP B (15M) / STEP C (5M): the five shared sub-checks, X-of-5.

    The 15M/5M RSI check is the trend half of the note's condition (the LEVEL
    band already passed at 1H); everything else mirrors STEP A verbatim.
    """
    if snap is None:
        return {"passed": False, "passed_count": 0, "min_conditions": min_conditions,
                "checks": {}, "reasons": [RC_NO_SNAPSHOT], "indicators": None}
    checks = {
        "rsi_trend": snap["rsi"] > snap["rsi_prev"] if direction == "BUY"
                     else snap["rsi"] < snap["rsi_prev"],
        "ema21": _ema_ok(snap, direction),
        "vwap": _vwap_ok(snap, direction),
        "volume": _volume_ok(snap, direction),
        "bb": _bb_ok(snap, direction, near_pct),
    }
    passed_count = sum(1 for ok in checks.values() if ok)
    ok = passed_count >= min_conditions
    return {
        "passed": ok,
        "passed_count": passed_count,
        "min_conditions": min_conditions,
        "checks": checks,
        "reasons": [] if ok else [f"{RC_15M_CONDITIONS}:{passed_count}of5"
                                  if min_conditions == config.MIN_15M_CONDITIONS
                                  else f"{RC_5M_CONDITIONS}:{passed_count}of5"],
        "indicators": snap,
    }


# ------------------------------------------------------------- pipeline

def evaluate(frames: dict, cfg=config) -> dict:
    """Run STEP A -> B -> C over the {1h, 15m, 5m} frames dict.

    Returns the full verdict: direction, per-step results, sweep, snapshots,
    and — when any step fails — the explicit machine-readable reason codes
    (why_no_signals.py groups them under the "strategy core" layer). Pure.
    """
    htf_df = frames.get(cfg.TF_HTF)
    setup_df = frames.get(cfg.TF_SETUP)
    entry_df = frames.get(cfg.TF_ENTRY)

    htf_snap = compute_indicators(htf_df) if htf_df is not None else None
    if htf_snap is None:
        return {"passed": False, "direction": None, "reasons": [RC_NO_SNAPSHOT],
                "steps": {}, "sweep": None,
                "snaps": {"1h": None, "15m": None, "5m": None}}

    sweeps = {"BUY": detect_sweep(htf_df, "BUY"), "SELL": detect_sweep(htf_df, "SELL")}
    # both directions tried; first fully-passing wins
    step_a = None
    for d in _DIRECTIONS:
        result = evaluate_1h_from(htf_snap, sweeps, d, cfg)
        if result["passed"]:
            step_a = result
            break
        if step_a is None or len(result["reasons"]) < len(step_a["reasons"]):
            step_a = result
    if not step_a["passed"]:
        return {"passed": False, "direction": None, "reasons": step_a["reasons"],
                "steps": {"1h": step_a}, "sweep": step_a["sweep"],
                "snaps": {"1h": htf_snap, "15m": None, "5m": None}}

    direction = step_a["direction"]
    setup_snap = compute_indicators(setup_df) if setup_df is not None else None
    step_b = evaluate_ltf(setup_snap, direction, cfg.BB_NEAR_PCT,
                          cfg.MIN_15M_CONDITIONS, cfg)
    if not step_b["passed"]:
        return {"passed": False, "direction": direction, "reasons": step_b["reasons"],
                "steps": {"1h": step_a, "15m": step_b}, "sweep": step_a["sweep"],
                "snaps": {"1h": htf_snap, "15m": setup_snap, "5m": None}}

    entry_snap = compute_indicators(entry_df) if entry_df is not None else None
    step_c = evaluate_ltf(entry_snap, direction, cfg.BB_NEAR_PCT,
                          cfg.MIN_5M_CONDITIONS, cfg)
    if not step_c["passed"]:
        return {"passed": False, "direction": direction, "reasons": step_c["reasons"],
                "steps": {"1h": step_a, "15m": step_b, "5m": step_c},
                "sweep": step_a["sweep"],
                "snaps": {"1h": htf_snap, "15m": setup_snap, "5m": entry_snap}}

    return {"passed": True, "direction": direction,
            "reasons": [], "steps": {"1h": step_a, "15m": step_b, "5m": step_c},
            "sweep": step_a["sweep"],
            "snaps": {"1h": htf_snap, "15m": setup_snap, "5m": entry_snap}}
