# Crypto Signal Bot

Signals-only crypto market-intelligence bot. Scans the full Binance **USDT-M
futures** market every 5 minutes, **24 hours a day** (288 scans/day, no session
window). A **deterministic price-action & market-context core** decides
**LONG / SHORT / NO_TRADE** (1H bias → 15M setup → 5M entry): market structure,
support/resistance zones, liquidity sweeps and a risk/reward gate decide —
indicators are only **secondary confirmation**. It sets entry/SL/TP/RR, alerts
BUY/SELL on Telegram (**NO_TRADE stays silent**), and logs every signal to CSV.
Since **Phase 4**, the self-hosted **Ollama LLM is the primary decision maker**
for every strategy-core-passing candidate: it picks the direction and its own
SL/TP, and Python validates only data validity, level geometry, stop width and
minimum RR (see the AI layer section below).

**This bot never trades.** No Binance API keys, no order endpoints — public
market data only.

> The decision hierarchy is defined in **[price_action_spec.md](price_action_spec.md)**
> (the primary authority). The owner's handwritten indicator note lives in
> **[strategy_spec.md](strategy_spec.md)** and now documents the **secondary**
> confirmation layer. This README describes how the code currently implements it.

## Install & run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux: source .venv/bin/activate)
pip install -r requirements.txt

copy .env.example .env            # fill in TELEGRAM_TOKEN / TELEGRAM_CHAT_ID /
                                  # OPENROUTER_API_KEY (all optional for testing)

python main.py                    # production: APScheduler, 24/7 (IST)
python main.py --once             # DEMO/TEST: one full scan cycle right now
```

Without keys the bot still runs: the deterministic core decides regardless of
the LLM, so explanations are written by a local template (alert footer *"Decision
by deterministic core · explanation generated locally"*) and Telegram alerts are
logged instead of sent.

## AI layer (Phase 4 — the LLM decides; Python validates)

Since the Phase 4 alignment, the **LLM is the primary decision maker for every
strategy-core-passing candidate**. Each scan sends those candidates to the
model with the FULL factual bundle (no Python verdict leaked): coin, price,
entry, 1H zone top/bottom, complete sweep details, all 15M indicator values,
the last ten 5M RSI values with an explicit trend word, EMA21/VWAP/Bollinger
for all three timeframes, the 5-candle volume trend, recent swing high/low and
ATR. The model answers strict JSON (signal/SL/TP/RR/confidence/one-line
reason) under `response_format=json_object` with the tolerant parser as the
safety net.

- **Application contract (`decision.post_llm_validate`):** the model's verdict
  and its own SL/TP are used as-is iff the levels are geometrically valid,
  the stop is within `RISK_MAX_STOP_ATR` of ATR and the implied RR is at
  least `MIN_RR`. On any violation Python OVERRIDES the levels with its own
  risk-gate levels (never a quality re-veto); malformed data is the only
  hard veto. A model HOLD is logged, silent. `ai_used=TRUE` marks every row
  the model decided; the model's own levels are visible in the CSV.
- **Journal:** `ai_opinions.csv` rows record `deterministic_decision`,
  `ai_opinion`, `agreement` (`SIGNAL_PROPOSED` = the model traded where the
  deterministic core would not) and `final_decision` = what actually shipped.
  The journal is written in the critical path (the old background-only audit
  is superseded).
- **LLM unreachable → deterministic fallback:** the funnel-passing setup is
  never dropped; its CSV reason is tagged *"AI Unavailable — deterministic
  core decision stands"*.
- **Deadline:** the whole stage is bounded by `SCAN_DEADLINE_SECONDS`
  (`llm_verdicts` honours it; `AI_WORKERS` bounds concurrency at 3). With a
  selective funnel the candidate count is small (measured: 3 candidates in a
  21-second scan on relaxed demo knobs).
- **`--force-llm`** stays a separate LOG-ONLY test harness that bypasses the
  gates to show what the model would answer on arbitrary setups.
- **Provider:** whatever `AI_BASE_URL`/`AI_MODEL` point at (default
  `https://openrouter.ai/api/v1` + `nvidia/nemotron-3-ultra-550b-a55b:free`);
  `AI_MODEL_FALLBACK` is empty by default, then the local template.
- **One-by-one:** `AI_BATCH_ENABLED=False` — each candidate gets its own
  full-data request (`AI_WORKERS=3` bounded concurrency, `AI_BATCH_MAX=8` only
  applies when batching is switched on), so one malformed batch answer can
  never lose several setups at once.
- **Output contract:** the batch request sets `response_format={"type":"json_object"}`,
  so the provider is constrained to JSON rather than merely asked for it. A gateway
  that rejects the field is detected on its first 400 and the capability is switched
  off for the process (`ai_decision.provider_caps()`). Replies are still read
  tolerantly — prose around the JSON is skipped, ``` fences stripped, a reply cut
  off by `max_tokens` salvaged back to its last complete verdict — and a truncated
  answer is retried with a correction turn and a larger token budget
  (`AI_MAX_TOKENS_RETRY_CAP`) instead of the same request three times. Nothing is
  ever invented for a setup the model did not answer.
- **`ai_used`** in `signals_log.csv` means "a model verdict replaced the
  deterministic one", which scheduled scans never do: `ai_used=False` on every row
  is the designed value, not an outage. The audit outcome is the `AI AUDIT …`
  journal line (`x/y answered | LONG n SHORT n NO_TRADE n | applied=never`) and
  `ai_opinions.csv`.
- **Retry:** 3 attempts per model with exponential backoff on 429 / 5xx / timeout
  / malformed JSON before falling to the next model.
- `AI_MAX_TOKENS = 8000` (with the truncation-salvage retry escalating to
  `AI_MAX_TOKENS_RETRY_CAP=12000`). Reasoning mode stays **off** on purpose — with a token cap the
  model spends the budget thinking and never emits the JSON.
- Per-IST-day budget accounting (`AI_DAILY_BUDGET = 0` = **unlimited**, the
  right value for a self-hosted Ollama; set a number only for metered cloud
  providers), surfaced in `/status`, with a one-time Telegram notice when the
  budget is exhausted (the notice fires
  from the worker's own finally-block, because a spent budget also spends retries —
  without it the audit file just fills with `UNAVAILABLE` rows and nothing explains
  why).

## DEMO / TEST modes (safe, no trading)

| Command | What it does |
|---|---|
| `python main.py --once` | One real production scan cycle on live data |
| `python scripts/e2e_demo.py [SYMBOL]` | Forces the gates open on real candles and runs decision → guard → alert → CSV end-to-end (writes `signals_log_demo.csv`) |
| `python scripts/funnel_check.py` | Live funnel stats: coins → sweeps → 1H → 15M → 5M |
| `python scripts/why_no_signals.py [--all] [--fix target]` | Offline: turns `signals_log.csv`'s `no_trade_reason` rows into a per-scan blocker histogram and shows how many alerts each layer would free |
| `python scripts/target_fix_compare.py [--max N] [--log PATH]` | Read-only: runs the real decision core twice on identical candles with the target scan off then on, and counts what the fix actually buys |
| `python scripts/schedule_check.py` | Prints all scheduler fire times for the next 24h (expects 288 scans/day, every 5 minutes at :15 seconds) |
| `python backtest.py --horizon-hours 24` | Backtests `signals_log.csv` against real Binance history → `backtest_report.txt` |
| `python backtest.py --strategy BTC/USDT:USDT` | Replays the **live decision core** over recent 1H/15M/5M history (look-ahead-safe) → `backtest_strategy_report.txt` |
| `python -m pytest tests/ -q` | Full unit test suite (no network needed) |

## Tuning the floors without editing code

Every strategy floor is `.env`-overridable (`MIN_RR`, `QUALITY_MIN`,
`QUALITY_PRIMARY_FLOOR`, `ALERT_QUALITY_MIN`, `ALERT_TIER_*`, `MIN_SCORE_1H/15M/5M`,
`MIN_CONFLUENCE`, `RISK_MIN_TARGET_ATR`, `RISK_MAX_STOP_ATR`). The **defaults are the
spec values**, and `config.check_config_warnings()` prints a `CONFIG:` line at startup
for an override that empties a rule (out-of-band floor, `ALERT_QUALITY_MIN` above
`QUALITY_MIN`, or RR enforced nowhere) — so an experiment is visible in the journal
instead of looking like a silent regression. A non-numeric override raises at startup
rather than being ignored.

## Required `.env` variables

Values are read raw — no quotes, no angle brackets, and **never paste a URL out of
a chat window or a markdown file** (`[https://host/v1](https://host/v1)` is a link,
not a URL, and it makes every AI call fail while the retry ladder burns the daily
budget; startup prints a `CONFIG:` line for it now). Verify what the file actually
contains: `grep -n AI_ .env | cat -A`.

```
TELEGRAM_TOKEN=<from @BotFather>        # optional; alerts logged if missing
TELEGRAM_CHAT_ID=<your chat id>         # REQUIRED for /scan_on, /scan_off (comma-separated for more)

# ---- Ollama (self-hosted LLM — the PRIMARY decision maker, Phase 4) ----
OLLAMA_BASE_URL=http://localhost:11434/v1   # optional; set BOTH to enable
OLLAMA_MODEL=qwen3:8b                       # any local tag; OpenAI-compatible endpoint

OPENROUTER_API_KEY=<from openrouter.ai>     # optional cloud fallback; AGENTROUTER_API_KEY also accepted
AI_BASE_URL=https://openrouter.ai/api/v1    # optional; any OpenAI-compatible provider
AI_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free # optional override
AI_MODEL_FALLBACK=minimax/minimax-m3:free,z-ai/glm-5.2:free # optional comma-separated ladder
AI_JSON_MODE=1                              # optional; 0 = don't send response_format
AI_MAX_TOKENS_RETRY_CAP=12000               # optional; ceiling for a truncated reply
ACCOUNT_BALANCE=1000                    # optional; USDT balance for position sizing
RISK_PER_TRADE_PCT=2.0                  # optional; max % risk per trade
LOG_LEVEL=INFO                          # optional
```

A configured Ollama provider LEADS the ladder (`ollama` first, cloud providers
after). Local endpoints are exempt from `AI_DAILY_BUDGET` (retry/backoff/
timeout still apply); cloud providers keep the `MIN_CONFLUENCE_FOR_AI`
eligibility floor, while the local provider receives every funnel-passing
candidate (`MIN_CONFLUENCE_FOR_AI_LOCAL=0`).

## Pipeline (every 5 minutes, 288 scans/day)

```
ALL USDT-M FUTURES PAIRS (fetch_tickers, dynamic)
→ drop stablecoin / leveraged-token bases (EXCLUDED_BASES + UP/DOWN/BULL/BEAR)
→ 24h volume ≥ $5M USDT (owner alignment; was $50M)
→ 1H structure funnel → duplicate guard checked BEFORE 15M/5M fetches
   (cooling-down coins cost zero LTF fetches)
→ retain 1H/15M/5M frames → decision.decide() — the deterministic core:
    1 market structure · 2 support/resistance zones · 3 liquidity & sweeps
    4 price action & volume · 5 trendlines · 6 multi-TF (1H→15M→5M, mandatory)
    7 OI + funding (contextual) · 8 risk/reward gate (mandatory): SL off structure,
    TP off the nearest *usable* opposing zone — one straddling the entry, or whose
    padded level would sit on the wrong side of it, is skipped, not fatal
   9 RSI/EMA/VWAP/Bollinger — SECONDARY confirmation, bounded, never a gate
  → LONG / SHORT / NO_TRADE + entry/SL/TP/RR + setup_quality + evidence
    NO_TRADE is valid & PREFERRED when confluence is thin (a trade is never forced)
→ rank funnel-passing survivors by setup_quality; each goes to the Ollama/LLM
   decision stage (full factual prompt, strict JSON verdict + its own SL/TP);
   post_llm_validate applies it (data/geometry/stop-width/MIN_RR overrides only)
   → LLM unreachable → deterministic decision stands, tagged "AI Unavailable"
→ journal every candidate to ai_opinions.csv (final_decision = what shipped)
→ funding-rate-aware leverage + risk-based position size (2% of ACCOUNT_BALANCE)
→ duplicate guard record (20 min per coin; expired entries pruned every scan —
   no session reset, the bot runs 24/7)
→ Telegram alert (BUY/SELL only — NO_TRADE → HOLD stays silent)
→ signals_log.csv (append-only, 30 columns); a blocked setup logs a silent HOLD
   row at most once per HOLD_LOG_COOLDOWN_MIN (30 min) per coin+reason, so the
   journal and the CSV keep the audit trail without repeating it 60× a session
```

### Liquidity sweeps (primary) & the no-sweep confidence cap

- **Liquidity & sweeps are primary evidence (layer 3):** a long needs a
  **sell-side sweep + reclaim + mandatory confirmation candle** (mirror for a
  short); a touch alone is not an entry (`liquidity.py`, `LIQ_CONFIRM_REQUIRED`).
- The legacy graded **zone/sweep** scorer in `scoring.py` is retained as part of
  the **secondary** indicator layer: with no fresh sweep the alert's confidence
  is **capped at 59** (`NO_SWEEP_CONFIDENCE_CAP`, below the HIGH tier) and labelled
  "Not detected (confidence capped)".

## Timeliness (never late)

- **One** cron job, `hour=*, minute=*/5, second=15` — the `:15` offset guarantees
  the just-closed 5M candle is published before the scan reads it, and a single
  job means no hour-boundary overlap. `max_instances=1`, `misfire_grace_time=120`,
  `coalesce=True` are set explicitly.
- **OHLCV TTL cache** keyed `(symbol, timeframe)` — 1H refetches hourly, 15M every
  15 min, 5M every scan — plus **concurrent fetching** (8 workers, shared token
  bucket under Binance's rate limit).
- **Hard per-scan deadline** (`SCAN_DEADLINE_SECONDS = 285`): past it the scan stops
  AI work and suppresses undecided trade opinions — so a scan can never
  bleed into the next 5-minute slot.

## Example Telegram alert

```
🟢 BUY SIGNAL — BTC/USDT  |  LONG 📈
━━━━━━━━━━━━━━━━━━
📊 RSI Bounce at 50 detected ✅

🧭 Market Context (primary)
├ Structure: uptrend/bullish [BOS-up]
├ HTF bias: bullish
├ Target zone: major@63280
└ Liquidity: sell-side sweep reclaimed

📊 Indicators (secondary confirmation)
├ RSI: 58.1 → 63.4 📈
├ EMA21: ✅ Above
├ VWAP: ✅ Above
└ Volume: 1.8x avg 💹

🌊 Liq Sweep: bullish detected

💰 Trade Levels
├ Entry:  63058.5
├ SL:     62946.6
├ TP:     63282.3
└ RR:     1:2.00

🔥 Confidence: HIGH
🎯 Confluence: 78/100  (1H 82 · 15M 74 · 5M 76)
📝 Sell-side sweep below 62946 reclaimed; HTF bullish BOS; volume rising
━━━━━━━━━━━━━━━━━━
⏰ 24/7 (IST)  |  1H → 15M → 5M
🧠 Decision by deterministic core · explanation generated locally
```

The **Market Context** block (structure / HTF bias / target zone / liquidity) is
the *primary* evidence; indicators sit below it, explicitly labelled *secondary*.
When a real LLM writes the prose the footer becomes `🤖 AI: <model>`.

## signals_log.csv

Append-only, **30 columns**: the original 20 (`timestamp, coin, signal, entry, SL,
TP, RR, leverage, position_size, funding_rate, confidence, confluence, score_1h,
score_15m, score_5m, sweep, sweep_age, rsi_bounce, reason, ai_used`) plus
`liquidation` and 8 structure-first evidence columns (`decision, setup_quality,
htf_bias, structure, sr_zone, liquidity, no_trade_reason, data_warnings`), and
`signal_id` for cross-referencing `ai_opinions.csv`. The row is built generically
from `config.CSV_COLUMNS`, so a field can never be silently dropped. At startup
`logger.migrate_csv_header()` archives any file with a stale header to
`signals_log.csv.vN.bak` and starts a fresh, correctly-headed file — history is
preserved, never overwritten.

## Files

`main.py` (entry + scheduler + scan loop), `config.py`, `scanner.py` (scan +
OHLCV cache + concurrency + exclusions + OI history). **Decision core:**
`decision.py` (the brain), `market_structure.py`, `support_resistance.py`,
`liquidity.py`, `price_action.py`, `trendlines.py`, `mtf.py`, `futures_context.py`,
`setup_quality.py`, `risk_gate.py`. **Secondary/support:** `indicators.py`,
`scoring.py` (secondary indicator-confirmation scorer), `filter_1h.py`
(+ sweep detection), `filter_15m.py`, `filter_5m.py` (per-TF feature extractors),
`ai_decision.py` (batch + retry + budget), `fallback.py` (local explanation),
`duplicate_guard.py`, `alerts.py`, `telegram_bot.py` (chat listener),
`chat_assistant.py`, `logger.py`, `backtest.py` (logged-signal + `--strategy`
replay), `requirements.txt`. Plus `tests/` (533 tests) and `scripts/`.
Spec docs live in the repo root; **[price_action_spec.md](price_action_spec.md)**
is the primary authority, **[strategy_spec.md](strategy_spec.md)** the secondary
indicator layer, and `memory.md` tracks progress.

## Accuracy notes

- pandas-ta does ALL indicator math; `bbands(ddof=0)` matches TradingView's
  population-std bands.
- Each timeframe fetches 50 strategy candles + 250 warm-up candles so the Wilder
  RSI/EMA recursions converge to TradingView values (verified <1e-6 in
  `tests/test_indicators.py`).
- Only closed candles are used — the forming candle is dropped at fetch time.

## Safety

Signals only. No `create_order` anywhere, no Binance keys (public mode), no web
dashboard, no multi-user. The scheduler runs 24/7 — every scan uses only public
market-data endpoints plus the configured AI endpoint; there is no session
window and no quiet hours.
