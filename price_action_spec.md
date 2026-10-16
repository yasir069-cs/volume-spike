# Price-Action & Market-Context Specification — PRIMARY AUTHORITY

> This file is the **primary source of truth** for *what decides a trade*.
>
> The bot decides on **market structure and market context first**; technical
> indicators (RSI, EMA21, VWAP, Bollinger, MACD) are **secondary confirmation
> only** and can never trigger or veto a trade on their own. The owner's
> handwritten indicator note lives in [`strategy_spec.md`](strategy_spec.md) and
> now documents that **secondary** layer.
>
> When code and this file disagree, **this file is right and the code is wrong**
> for the *decision hierarchy*. Concrete thresholds are named constants in
> [`config.py`](config.py); the values quoted here are current-as-written and
> the constant name is the authority if they ever drift.

## Design principle

> **`NO_TRADE` is a valid — and preferred — outcome whenever multi-factor
> confluence is insufficient. The system never forces a trade.**

A coin below VWAP can still go **LONG** if structure plus a reclaimed liquidity
sweep say so. A textbook indicator alignment printing into the underside of a
major resistance with poor reward returns **NO_TRADE**. Indicators inform; they
do not command.

## The decision hierarchy (Phase 2 alignment — indicator funnel primary)

> **Phase 2 change:** the structure-first hierarchy below was demoted. The
> owner's handwritten indicator funnel (`strategy_core.py`) is the PRIMARY
> gate; the former primary layers now only SCORE and provide LEVELS, with a
> single surviving HTF veto. This section documents the current contract.

### Primary gate — `strategy_core.evaluate()` (called first by `decision.decide`)

| Step | Timeframe | Rule | Fails with |
|---|---|---|---|
| A | 1H context | zone in the favourable 30% (`ZONE_FULL_PCT`) · RSI 50–70 rising / 35–50 falling (`RSI_*_FULL_*`) · EMA21 side · VWAP side · volume increase · near/touching favourable Bollinger band (`BB_NEAR_PCT_1H`) · aligned liquidation sweep within `SWEEP_MAX_AGE_CANDLES` (5) candles when `REQUIRE_1H_SWEEP` — ALL must hold and they define the direction | `1h_zone`, `1h_rsi_band`, `1h_rsi_trend`, `1h_ema21`, `1h_vwap`, `1h_volume`, `1h_bb`, `1h_sweep` |
| B | 15M confirmation | the five shared checks (RSI trend / EMA21 / VWAP / volume / BB side) — need `MIN_15M_CONDITIONS` of 5 (default 4) | `15m_conditions:Xof5` |
| C | 5M entry | the same five checks — need `MIN_5M_CONDITIONS` of 5 (default 3) | `5m_conditions:Xof5` |

A STEP A failure skips the symbol before any 15M/5M fetch is paid for.

### Demoted layers (scoring + levels only — they can no longer return NO_TRADE)

| Layer | Module | Remaining role |
|---|---|---|
| Market structure | `market_structure.py` | feeds `setup_quality`; supplies the structural SL (last swing) |
| Support / Resistance | `support_resistance.py` | supplies the opposing-zone TP; scoring context |
| Liquidity & sweeps | `liquidity.py` | confirmation state feeds scoring and the LLM facts |
| Price action & volume | `price_action.py` | scoring signals |
| Trendlines | `trendlines.py` | scoring only |
| Multi-timeframe | `mtf.py` | scoring evidence; its `counter_htf` VETO is removed |
| Direction gate | `direction_gate.py` | its `conflict` VETO is removed — disagreement is now a bounded scoring factor |
| Futures context | `futures_context.py` | contextual scoring, safe-degrading (unchanged) |
| Setup quality | `setup_quality.py` | RANKS candidates + sets the alert tier; `insufficient_primary_evidence` / `low_setup_quality` blockers are removed |

**The one surviving HTF veto:** a BUY is blocked only when the 1H structure is a
CONFIRMED downtrend WITH a bearish BOS (`htf_confirmed_downtrend`); mirror for a
SELL (`htf_confirmed_uptrend`).

**Still mandatory:** the risk/reward gate — `MIN_RR` (default 1.5, hard),
`no_structure_stop` (SL = last swing level padded by `RISK_SL_BUFFER_ATR` x ATR),
`stop_too_wide`, `no_atr`, `excessive_spread`, `into_opposing_zone`.
**Phase 5:** the TP comes from the next usable opposing zone; when none
qualifies it falls back to the FIXED 1:2 target (TP = entry +/- 2x the stop
distance, labelled `fixed-1:2`) instead of `NO_TRADE` — `no_clear_target`
survives only when no stop exists either, and `target_too_close` only when
even the fallback violates the reward floor (pathological ultra-tight stop).
Every BUY/SELL alert also states one line of WHICH conditions passed
(`Strategy Gates`: 1H zone/RSI/EMA21/VWAP/volume/BB + sweep age, 15M X-of-5,
5M X-of-5) and shows the visible NORMAL/HIGH/STRONG tier with
`ALERT_QUALITY_MIN` (50) as the documented alert floor.

## How the core combines them — [`decision.decide()`](decision.py)

```
frames = { "1h": HTF, "15m": setup, "5m": entry }   # config.TF_HTF / TF_SETUP / TF_ENTRY

1. Frame adequacy      each TF >= _MIN_FRAME (8) closed candles, else NO_TRADE("insufficient_data")
2. Structure x3 + MTF  market_structure.analyze per TF -> mtf.combine -> htf_bias, direction, counter_htf
   └─ direction is None (no HTF bias / range)          -> NO_TRADE("no_directional_bias")
3. Detectors           support_resistance / liquidity / price_action / trendlines / futures_context
4. Indicators (sec.)   scoring.indicator_confirmation(snap, direction) -> {score, agrees, notes}
5. Setup quality       setup_quality.score(...) -> primary (0-100) + bounded indicator term
   └─ primary < QUALITY_PRIMARY_FLOOR (45)              -> NO_TRADE("insufficient_primary_evidence")
   └─ quality  < QUALITY_MIN (50)                       -> NO_TRADE("low_setup_quality")
6. MTF alignment       MTF_REQUIRE_HTF_ALIGN and entry opposes HTF -> NO_TRADE("counter_htf")
7. Risk/reward gate     risk_gate.evaluate(...) -> structure SL, target off nearest opposing zone, RR
   └─ rr < MIN_RR (1.5) / stop > RISK_MAX_STOP_ATR (3.0)
      / opposing zone < RISK_MIN_TARGET_ATR (1.0) / spread > RISK_MAX_SPREAD_PCT -> NO_TRADE(reason)

decision = direction  iff  no reason accumulated,  else  NO_TRADE
```

The returned dict carries: `decision`, `direction`, `setup_quality`/`confidence`,
`htf_bias`, `mtf`, `structure`, `sr`, `liquidity`, `price_action`, `trendline`,
`futures`, `indicators` (secondary), `quality`, `risk`, `entry`, `sl`, `tp`, `rr`,
`no_trade_reasons`, and `data_warnings`.

### Setup-quality weights (primary-weighted; indicators bounded)

Primary evidence weights (`config.QUALITY_W_*`, sum = 100):

| Structure | S/R | Liquidity | Price action | MTF | Trendline | Futures |
|---|---|---|---|---|---|---|
| 25 | 20 | 20 | 15 | 10 | 5 | 5 |

Indicators enter only as a **bounded secondary term** on top of the primary
score and are floored out entirely below `QUALITY_PRIMARY_FLOOR` — they cannot
manufacture a signal the primary evidence does not support.

## Timeframe roles (reused 1H / 15M / 5M — configurable)

- **1H — `config.TF_HTF`:** higher-timeframe bias / market context.
- **15M — `config.TF_SETUP`:** structure, S/R, liquidity, trendline validation.
- **5M — `config.TF_ENTRY`:** price-action trigger; its **close is the entry**.

No 4H fetch — the three existing frames are retained through `run_scan` and fed
to the core, so there is no added fetch cost.

## Who decides vs who explains

- **The deterministic Python core decides.** Same rules run in **live and
  backtest** (`backtest.py --strategy` steps history bar-by-bar through the very
  same `decision.decide`, look-ahead-safe).
- **The LLM explains only.** It receives the finished decision and writes prose;
  a slow, failed, or disabled LLM is **cosmetic** and can never change or drop a
  signal. When the LLM is unavailable, [`fallback.py`](fallback.py)
  `explanation_fallback()` writes a deterministic local template and the alert is
  tagged *"explanation generated locally"* (not the legacy "AI unavailable").

## Crypto-futures data (contextual, safe-degrading)

Verified keyless via ccxt: Open-Interest history + funding rates are fetched
(`OI_FETCH_ENABLED`, `OI_HISTORY_TIMEFRAME=5m`, `OI_HISTORY_LIMIT=24`).
Market-wide liquidation clusters are **not** public keyless — the candle-based
liquidity sweep is the actionable proxy. Basis / long-short ratio are left as
config-flagged, safe-degrading slots. Missing data → `futures.available = False`
+ a `data_warnings` entry; the decision still returns.

## NO_TRADE reasons (exact strings)

`insufficient_data` · `no_directional_bias` · `counter_htf` ·
`directional_conflict` · `insufficient_primary_evidence` · `low_setup_quality` ·
plus risk-gate reasons (`no_atr`, `no_structure_stop`, `no_clear_target`,
`stop_too_wide`, `target_too_close`, `poor_rr`, `into_opposing_zone`,
`excessive_spread`).
A `NO_TRADE` maps to **HOLD** at the signal boundary: logged, silent — no alert,
no cooldown.

## Out of scope / invariants (unchanged)

**Signals only.** The bot never places a trade (`create_order`), holds no
exchange API keys, and uses only public market data. Existing risk &
position-sizing limits are respected, never bypassed. Look-ahead / future
leakage is disallowed everywhere: detectors see only **closed candles ≤ the
decision bar**.
