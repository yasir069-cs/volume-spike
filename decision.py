"""Decision core — the single source of truth for LONG / SHORT / NO_TRADE.

Phase 2 alignment: **the owner's indicator funnel decides.**

`strategy_core.evaluate()` is the PRIMARY gate, run first over the frames:

    STEP A  1H context pass/fail  -> direction (BUY/SELL) or explicit skip
    STEP B  15M confirmation      -> need MIN_15M_CONDITIONS of 5 (default 4)
    STEP C  5M entry conditions   -> need MIN_5M_CONDITIONS of 5 (default 3)

When STEP A fails the symbol is rejected before the demoted detectors even
run — in the live scan that also means before any 15M/5M fetch is paid for.

The former structure-first layers keep exactly two jobs plus ONE surviving
veto (the "demotion contract"):

  * SCORING — market structure, S/R, liquidity, price action, trendlines,
    MTF and futures context feed `setup_quality.score()` (0-100). The score
    RANKS candidates and sets the alert tier; it can no longer return
    NO_TRADE by itself (`insufficient_primary_evidence` / `low_setup_quality`
    no longer exist as blockers — that pair alone blocked 37 of 41 live
    decisions on 2026-09-16).
  * LEVELS — `risk_gate` still builds the structural SL / opposing-zone TP
    and stays a MANDATORY hard gate (MIN_RR, stop width, spread, no-ATR);
    `no_clear_target` remains a blocker until Phase 5 adds the fixed 1:2
    fallback target.
  * ONE HTF veto — a BUY is blocked only when the 1H structure shows a
    CONFIRMED downtrend (trend == "downtrend") WITH a bearish BOS; mirror
    for a SELL. The old `counter_htf` (entry-TF-vs-HTF-bias) and
    `directional_conflict` (direction-gate) vetoes are GONE as blockers:
    direction_gate's disagreement now only discounts the score (its factor
    feeds setup_quality), and mtf's counter-HTF note is scoring evidence.
    Removals are documented in the Phase 2 report.

`decide()` is pure over the frames handed to it (a dict keyed by timeframe
label; roles resolved from config.TF_HTF / TF_SETUP / TF_ENTRY). In live use
the frames already contain only closed candles; in backtest they are sliced
to the decision bar — so there is no look-ahead either way. The same
function runs in both, so a backtest replays exactly what live would have
decided.

**NO_TRADE is a first-class result.** Every rejection carries explicit
machine-readable reasons (`1h_zone`, `15m_conditions:3of5`, `poor_rr`, …).
Missing OI/funding degrade to data warnings, not failures.
"""
import logging
from typing import Optional

import config
import direction_gate
import futures_context
import liquidity as liquidity_mod
import market_structure
import mtf as mtf_mod
import price_action as price_action_mod
import risk_gate
import scoring
import setup_quality
import strategy_core
import support_resistance
import trendlines as trendlines_mod

log = logging.getLogger("decision")

_MIN_FRAME = config.STRUCT_PIVOT_LEFT + config.STRUCT_PIVOT_RIGHT + 2


def _blank(decision: str, reasons: list, warnings: list, **extra) -> dict:
    """Assemble a structured result, filling absent evidence with None/defaults."""
    out = {
        "decision": decision,
        "direction": None,
        "setup_quality": 0.0,
        "confidence": 0.0,
        "primary": 0.0,
        "htf_bias": "neutral",
        "mtf": None,
        "structure": None,
        "sr": None,
        "liquidity": None,
        "price_action": None,
        "trendline": None,
        "futures": None,
        "indicators": None,
        "quality": None,
        "risk": None,
        "entry": None,
        "sl": None,
        "tp": None,
        "rr": None,
        "no_trade_reasons": reasons,
        "data_warnings": warnings,
    }
    out.update(extra)
    return out


def _htf_bos_veto(direction: str, htf_struct: dict) -> Optional[str]:
    """The single surviving HTF veto.

    A BUY is blocked only when the 1H is in a CONFIRMED downtrend AND has
    printed a bearish BOS (continuation break) — i.e. the higher timeframe
    is actively breaking DOWN, not merely ranging. Mirror for a SELL.
    Returns the reason code or None.
    """
    if direction == "LONG":
        if (htf_struct.get("trend") == "downtrend"
                and (htf_struct.get("bos") or {}).get("dir") == "bearish"):
            return "htf_confirmed_downtrend"
    else:
        if (htf_struct.get("trend") == "uptrend"
                and (htf_struct.get("bos") or {}).get("dir") == "bullish"):
            return "htf_confirmed_uptrend"
    return None


def decide(frames: dict, funding_rate: Optional[float] = None,
           oi_df=None, cfg=config, symbol: Optional[str] = None) -> dict:
    """Return the full structured decision for one symbol.

    `frames` maps timeframe labels (e.g. "1h", "15m", "5m") to OHLCV frames.
    Returns a dict whose "decision" is LONG, SHORT or NO_TRADE plus all the
    evidence and (for NO_TRADE) the exact reasons. `symbol` is optional and
    only used for penalty logging.
    """
    warnings: list[str] = []
    htf_df = frames.get(cfg.TF_HTF)
    setup_df = frames.get(cfg.TF_SETUP)
    entry_df = frames.get(cfg.TF_ENTRY)

    for label, df in ((cfg.TF_HTF, htf_df), (cfg.TF_SETUP, setup_df), (cfg.TF_ENTRY, entry_df)):
        if df is None or len(df) < _MIN_FRAME:
            warnings.append(f"insufficient_frame:{label}")
    if warnings:
        return _blank("NO_TRADE", ["insufficient_data"], warnings)

    # ---- PRIMARY GATE: the owner's indicator funnel (STEP A -> B -> C) ----
    core = strategy_core.evaluate(frames, cfg)
    if not core["passed"] or core["direction"] is None:
        return _blank("NO_TRADE", list(core["reasons"]) or ["strategy_core_reject"],
                      warnings, snaps=core.get("snaps") or {},
                      core=core)
    # vocabulary boundary: the funnel speaks BUY/SELL (the note's words), the
    # decision core and every downstream layer speak LONG/SHORT.
    direction = "LONG" if core["direction"] == "BUY" else "SHORT"

    # ---- demoted detectors: scoring evidence + risk-gate levels ----
    htf_struct = market_structure.analyze(htf_df, cfg)
    setup_struct = market_structure.analyze(setup_df, cfg)
    entry_struct = market_structure.analyze(entry_df, cfg)
    mtf = mtf_mod.combine(htf_struct, setup_struct, entry_struct, cfg)
    sr = support_resistance.analyze(setup_df, cfg)
    liq = liquidity_mod.analyze(setup_df, cfg)
    pa = price_action_mod.analyze(entry_df, cfg)
    tl = trendlines_mod.analyze(setup_df, cfg)
    futures = futures_context.interpret(oi_df, funding_rate, entry_df, cfg)
    warnings.extend(futures.get("warnings", []))

    entry = float(entry_df["close"].iloc[-1])
    atr = setup_struct["atr"] or sr["atr"]

    common = {"mtf": mtf, "structure": setup_struct, "sr": sr, "liquidity": liq,
              "price_action": pa, "trendline": tl, "futures": futures,
              "htf_bias": mtf["htf_bias"], "entry": round(entry, 8),
              "core": core, "snaps": core.get("snaps") or {}}

    # ---- the ONE surviving HTF veto ----
    reasons: list[str] = []
    veto = _htf_bos_veto(direction, htf_struct)
    if veto:
        reasons.append(veto)
        log.info("%s: %s blocked by the single HTF veto — 1H %s",
                 symbol or "setup", direction, veto)

    # ---- direction gate: SCORING PENALTY ONLY (its veto was removed) ----
    snap = core["snaps"].get("15m")
    htf_snap = core["snaps"].get("1h")
    entry_snap = core["snaps"].get("5m")
    indicator_conf = scoring.indicator_confirmation(snap, direction, cfg)
    dir_check = direction_gate.confirm(direction, htf_snap, snap, entry_snap,
                                       setup_struct, liq, mtf, cfg)
    if dir_check["status"] == "conflict":
        log.info("%s: %s direction CONFLICT — now a scoring penalty (%s), not a veto",
                 symbol or "setup", direction, "; ".join(dir_check["reasons"]))

    # ---- mandatory risk/reward gate (hard; levels for the alert) ----
    risk = risk_gate.evaluate(direction, entry, setup_struct, sr, atr, cfg)

    # ---- setup quality: RANKING + ALERT TIER only, never a veto ----
    quality = setup_quality.score(direction, setup_struct, sr, liq, pa, mtf, tl,
                                  futures, indicator_conf, cfg,
                                  htf_snap=htf_snap, risk=risk,
                                  direction_factor=dir_check["factor"],
                                  direction_reasons=dir_check["reasons"])
    if quality.get("penalties"):
        log.info("%s: %s penalized — %s (quality=%.1f, primary %.1f -> %.1f)",
                 symbol or "setup", direction, "; ".join(quality["penalties"]),
                 quality["quality"], quality.get("raw_primary", quality["primary"]),
                 quality["primary"])
    if not risk["ok"]:
        reasons.extend(risk["reasons"])

    decision = direction if not reasons else "NO_TRADE"
    if decision != "NO_TRADE":
        log.info("%s decided quality=%.1f rr=%s", decision, quality["quality"], risk["rr"])

    return {
        "decision": decision,
        "direction": direction,
        "setup_quality": quality["quality"],
        "confidence": quality["quality"],
        "primary": quality["primary"],
        "htf_bias": mtf["htf_bias"],
        "mtf": mtf,
        "structure": setup_struct,
        "sr": sr,
        "liquidity": liq,
        "price_action": pa,
        "trendline": tl,
        "futures": futures,
        "indicators": indicator_conf,
        "quality": quality,
        "direction_check": dir_check,
        "risk": risk,
        "core": core,
        "entry": round(entry, 8),
        "sl": risk["sl"],
        "tp": risk["tp"],
        "rr": risk["rr"],
        "no_trade_reasons": reasons,
        "data_warnings": warnings,
        # indicator snapshots by role, exposed so the LLM decision stage gets
        # the full structured data and a flipped direction can be re-scored
        # without re-fetching anything
        "snaps": {"1h": htf_snap, "15m": snap, "5m": entry_snap},
    }


def post_llm_validate(d: dict, direction: str, cfg=config,
                      llm_levels: Optional[dict] = None) -> dict:
    """Hard safety validation of an LLM-chosen direction (post-decision).

    The LLM decides independently from factual evidence (it never saw a Python
    verdict); this function then applies ONLY the gates that can never be
    bypassed:

      * data validity (indicator snapshots present, entry computable)
      * level safety, Phase 4 contract — when the model proposed its own
        levels they are USED as-is iff geometry is valid (BUY sl<entry<tp,
        SELL mirror), the stop is within RISK_MAX_STOP_ATR of ATR, and the
        level-implied RR is at least MIN_RR. On any of those violations (or
        missing/garbled levels) Python OVERRIDES with its own risk_gate
        levels and re-applies the deterministic risk reasons. The computed
        quality is retained as evidence and telemetry, but it is not a veto:
        the Ollama decision is primary after the basic prefilter.

    Returns a decision dict shaped like decide()'s: the chosen direction when
    every gate passes, else NO_TRADE with the exact gate reasons. Pure:
    reuses the evidence and snapshots stored in `d`.
    """
    if direction not in ("LONG", "SHORT"):
        out = dict(d)
        out.update({"decision": "NO_TRADE",
                    "no_trade_reasons": ["invalid_direction"]})
        return out

    snaps = d.get("snaps") or {}
    mtf = d.get("mtf") or {}
    structure = d.get("structure") or {}
    sr = d.get("sr") or {}
    liq = d.get("liquidity") or {}
    pa = d.get("price_action") or {}
    tl = d.get("trendline") or {}
    futures = d.get("futures") or {}
    entry = d.get("entry")
    atr = structure.get("atr") or (sr or {}).get("atr")
    htf_snap, setup_snap, entry_snap = snaps.get("1h"), snaps.get("15m"), snaps.get("5m")

    reasons: list[str] = []
    if entry is None or not htf_snap or not setup_snap:
        reasons.append("invalid_data")

    # indicator confirmation is direction-specific: recompute for the CHOSEN
    # direction instead of reusing the one computed for the core's direction
    indicator_conf = scoring.indicator_confirmation(setup_snap, direction, cfg)
    dir_check = direction_gate.confirm(direction, htf_snap, setup_snap,
                                       entry_snap, structure, liq, mtf, cfg)
    risk = risk_gate.evaluate(direction, entry, structure, sr, atr, cfg)
    quality = setup_quality.score(direction, structure, sr, liq, pa, mtf, tl,
                                  futures, indicator_conf, cfg,
                                  htf_snap=htf_snap, risk=risk,
                                  direction_factor=dir_check["factor"],
                                  direction_reasons=dir_check["reasons"])

    if not risk["ok"]:
        reasons.extend(risk["reasons"])

    # ---- Phase 4: LLM-proposed levels (validate-then-use, else override) ----
    # A violation never vetoes the trade by itself: Python OVERRIDES the bad
    # levels with its own risk_gate levels and the normal risk verdict then
    # governs. The override is recorded as a data warning + levels_note.
    sl, tp, rr = risk["sl"], risk["tp"], risk["rr"]
    levels_note = ""
    override_notes: list[str] = []
    llm_sl = llm_levels.get("sl") if llm_levels else None
    llm_tp = llm_levels.get("tp") if llm_levels else None
    if llm_sl is not None and llm_tp is not None and atr > 0:
        geom_ok = (llm_sl < entry < llm_tp if direction == "LONG"
                   else llm_tp < entry < llm_sl)
        stop_width = abs(entry - llm_sl)
        llm_rr = (abs(llm_tp - entry) / stop_width) if stop_width > 0 else 0.0
        if not geom_ok:
            override_notes.append("llm_levels_invalid_geometry")
            levels_note = "LLM levels discarded (geometry invalid) — Python levels used"
        elif stop_width > cfg.RISK_MAX_STOP_ATR * atr:
            override_notes.append("llm_stop_too_wide")
            levels_note = "LLM stop wider than RISK_MAX_STOP_ATR — Python levels used"
        elif llm_rr < cfg.MIN_RR:
            override_notes.append("llm_rr_below_min")
            levels_note = f"LLM levels RR {llm_rr:.2f} < MIN_RR — Python levels used"
        else:
            # the model's valid levels stand; the deterministic level-derived
            # reasons (poor_rr, no_clear_target, ...) do not veto them
            sl, tp, rr = llm_sl, llm_tp, round(llm_rr, 2)
            reasons = [r for r in reasons
                       if r not in ("poor_rr", "target_too_close", "no_clear_target",
                                    "no_structure_stop", "into_opposing_zone")]
            levels_note = "LLM levels validated and used"
        if override_notes:
            log.info("%s: %s", direction, "; ".join(override_notes) +
                     (" — Python levels used" if override_notes else ""))
    warnings = list(d.get("data_warnings") or []) + override_notes
    out = dict(d)
    out.update({
        "direction": direction,
        "decision": direction if not reasons else "NO_TRADE",
        "setup_quality": quality["quality"],
        "confidence": quality["quality"],
        "primary": quality["primary"],
        "quality": quality,
        "direction_check": dir_check,
        "risk": {**risk, "sl": (None if sl is None else round(sl, 8)),
                 "tp": (None if tp is None else round(tp, 8)),
                 "rr": rr, "levels_note": levels_note},
        "sl": None if sl is None else round(sl, 8),
        "tp": None if tp is None else round(tp, 8),
        "rr": rr,
        "no_trade_reasons": reasons,
        "data_warnings": warnings,
    })
    return out
