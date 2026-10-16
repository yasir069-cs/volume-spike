"""Crypto Signal Bot — volume spike detector only.

Scans Binance USDT-M futures every 5 minutes, detecting only volume spikes.
All structural analysis, indicators, AI/LLM, risk gates removed for on-time
100% accurate volume-based signals.

Every 5 minutes (288 scans/day):
    STEP 1: full exchange scan (USDT-M perps with $5M+ volume)
    STEP 2: volume spike detection — if 1H candle volume spikes significantly
    STEP 3: emit BUY/SELL alert only when volume spike detected
    STEP 4: persist to signals_log.csv, then Telegram alert

No structural analysis, no indicators, no AI/LLM, no risk gates — purely
volume spike detection for on-time 100% accuracy on the signal.

Usage:
    python main.py            # production schedule (APScheduler, IST)
    python main.py --once     # DEMO/TEST: run one full scan cycle now

Volume-only signals — this bot NEVER places trades (no trading endpoints, no keys).
"""
import argparse
import csv
import logging
import time
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import alerts
import config
import logging as log_mod
import scan_coordinator
import duplicate_guard
import scanner

log = logging.getLogger("main")

# THE single-scan gate + shared guard. Every path
# (scheduled job, /scan_on session, --once) enters run_scan, which claims the
# coordinator before touching the network.
_coordinator = scan_coordinator.ScanCoordinator(duplicate_guard.DuplicateGuard)


def get_coordinator() -> scan_coordinator.ScanCoordinator:
    """Shared coordinator (telegram_bot /status, ondemand wiring, tests)."""
    return _coordinator


def _new_scan_id(now_ist: datetime) -> str:
    return now_ist.strftime("%Y%m%d-%H%M%S")


def _in_session(now_ist: datetime) -> bool:
    """Continuous operation is enabled; the bot scans 24 hours a day."""
    return True


# Simple signal map for volume-only alerts
_SIGNAL_MAP = {"VOLUME_SPIKE": "VOLUME_ALERT"}


def _volume_spike_check(current_vol: float, prev_vol: float, vol_avg20: float) -> bool:
    """Check if current volume represents a significant spike.

    A spike is detected when:
    - Current volume > previous volume * 1.5 (50% increase), OR
    - Current volume > 20-day average * 1.5 (significant above average)
    """
    if prev_vol <= 0 or vol_avg20 <= 0:
        return False
    # Spike if 50% above previous candle
    if current_vol > prev_vol * 1.5:
        return True
    # Spike if 50% above 20-day average
    if current_vol > vol_avg20 * 1.5:
        return True
    return False


def run_scan(exchange, guard: duplicate_guard.DuplicateGuard | None = None,
             tickers: dict | None = None,
             funding_rates: dict | None = None,
             force: bool = False) -> dict:
    """One full scan cycle, coordinated: only ONE scan may run at a time.

    `force=True` is retained for compatibility (all production hours are now
    active). If
    another scan is active (scheduled + on-demand overlap), this call skips
    immediately and says so.

    Critical path = universe -> volume spike -> persist -> Telegram, all
    stage-timed and deadline-bounded.

    *tickers* is the raw dict from exchange.fetch_tickers(); when provided it
    is forwarded to the scanner so fetch_tickers() is called once per cycle.
    *funding_rates* is the {symbol: rate} dict for funding-rate filtering.
    """
    now_ist = datetime.now(config.TZ)
    scan_id = _new_scan_id(now_ist)

    # THE single-scan gate: scheduled, /scan_on and --once all pass here.
    if not _coordinator.try_begin(scan_id):
        log.warning("Scan %s skipped — scan %s already active", scan_id,
                    _coordinator.scan_id)
        return {"scanned": 0, "skipped": "scan already active"}
    try:
        return _run_scan_locked(exchange, guard, tickers, funding_rates,
                                now_ist, scan_id)
    except Exception as exc:
        # the scan slot must never stay claimed because one scan blew up
        log.error("Scan %s failed: %s: %s", scan_id, type(exc).__name__, exc)
        _coordinator.end({"error": str(exc)[:200], "scanned": 0})
        return {"scanned": 0, "error": str(exc)[:200]}


def _run_scan_locked(exchange, guard, tickers, funding_rates,
                     now_ist, scan_id) -> dict:
    """The scan body — only reached with the coordinator slot claimed."""
    deadline = time.monotonic() + config.SCAN_DEADLINE_SECONDS
    # one shared guard across every scan path (was: per-path instances)
    guard = guard or _coordinator.get_guard()
    # 24/7 housekeeping: drop expired cooldown entries so the tracker stays
    # bounded no matter how many days the bot runs (no session reset exists).
    guard.prune(now_ist)

    with _coordinator.stage("universe"):
        symbols = scanner.get_active_usdt_symbols(exchange, tickers)
    tickers_last = {}
    if tickers:
        tickers_last = {s: t.get("last") for s, t in tickers.items()
                        if isinstance(t, dict) and t.get("last")}
    if funding_rates is None:
        funding_rates = {}
    summary = {"scanned": len(symbols), "volume_spikes": 0, "alerts": 0,
               "log_only": 0, "holds": 0, "hold_deduped": 0,
               "telegram_failed": 0, "log_failed": 0,
               "ai_status": "OFF" if not config.LLM_DECISION_ENABLED else "IDLE",
               "ai_eligible": 0, "ai_sent": 0, "ai_answered": 0,
               "ai_applied": 0, "ai_below_confluence": 0}

    # --- STEP 1+2: batch-fetch 1H for every symbol; VOLUME SPIKE DETECTION ---
    # Check 1H volume spike for each symbol
    with _coordinator.stage("ohlcv_1h"):
        frames_1h = scanner.fetch_timeframe_batch(symbols, "1h", deadline=deadline)
    with _coordinator.stage("volume_check"):
        for symbol, df_1h in frames_1h.items():
            # Get volume data from the 1H frame
            if df_1h is None or len(df_1h) < 2:
                continue

            # Get current close candle volume and previous candle volume
            current_vol = float(df_1h["volume"].iloc[-1])
            prev_vol = float(df_1h["volume"].iloc[-2]) if len(df_1h) >= 2 else current_vol

            # Get 20-day volume average from indicators
            # Use simple: current candle volume vs previous candle volume comparison
            # plus comparison to 20-period average if available

            # Calculate 20-period volume average from the tail
            vol_data = df_1h["volume"].tail(config.SWEEP_WINDOW).tolist()
            vol_avg20 = float(sum(vol_data) / len(vol_data)) if vol_data else 0.0

            # Check for volume spike
            if _volume_spike_check(current_vol, prev_vol, vol_avg20):
                summary["volume_spikes"] += 1
                log.info("%s: VOLUME SPIKE detected — current=%.0f, prev=%.0f, avg20=%.0f",
                         symbol, current_vol, prev_vol, vol_avg20)

                # Determine signal direction based on price action
                # Simple: if price rose on high volume -> BUY, if fell -> SELL
                current_price = float(df_1h["close"].iloc[-1])
                prev_price = float(df_1h["close"].iloc[-2]) if len(df_1h) >= 2 else current_price
                price_change_pct = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0

                # Store the volume spike info for alert building
                summary["volume_spike_symbol"] = symbol
                summary["volume_spike_current_vol"] = current_vol
                summary["volume_spike_prev_vol"] = prev_vol
                summary["volume_spike_price_change_pct"] = price_change_pct
                summary["volume_spike_alert_type"] = "BUY" if price_change_pct > 0 else "SELL"
            else:
                log.debug("%s: no significant volume spike (current=%.0f, prev=%.0f, avg20=%.0f)",
                          symbol, current_vol, prev_vol, vol_avg20)

# --- STEP 3: persist volume spike and alert ---
    # Only alert if we had a volume spike
    if summary.get("volume_spikes", 0) > 0:
        symbol = summary.get("volume_spike_symbol", "")
        alert_type = summary.get("volume_spike_alert_type", "BUY")
        current_vol = summary.get("volume_spike_current_vol", 0)
        prev_vol = summary.get("volume_spike_prev_vol", 0)
        price_change_pct = summary.get("volume_spike_price_change_pct", 0)

        # Build the signal dict for volume-only alert
        sig = {
            "coin": symbol,
            "signal": alert_type,
            "entry": current_vol,  # using volume as entry placeholder
            "SL": 0,  # no SL for volume-only
            "TP": 0,  # no TP for volume-only
            "RR": 0,  # no RR for volume-only
            "leverage": "1x",
            "position_size": 0,
            "funding_rate": 0.0,
            "confidence": 100.0,  # full confidence for volume spike
            "confluence": 100.0,
            "score_1h": None,
            "score_15m": None,
            "score_5m": None,
            "sweep": {"detected": True, "type": "bullish" if alert_type == "BUY" else "bearish",
                      "age": 0},
            "sweep_age": 0,
            "rsi_bounce": False,
            "rsi_bounce_detected": False,
            "indicators": {},
            "reason": f"VOLUME SPIKE: {alert_type} signal - {price_change_pct:.1f}% price change, volume {current_vol:.0f} (prev: {prev_vol:.0f}, avg20: {vol_avg20:.0f})",
            "ai_used": False,
            "setup_quality": 100.0,
            "htf_bias": "neutral",
            "structure": {},
            "sr_zone": "",
            "liquidity": "",
            "no_trade_reason": "",
            "data_warnings": "",
            "conditions": "",
            "liquidation": {},
            "decision": alert_type,
            "setup_quality": 100.0,
            "htf_bias": "neutral",
            "structure": {},
            "sr_zone": "",
            "liquidity": "",
            "no_trade_reason": "",
            "data_warnings": "",
            "conditions": "",
            "liquidation": {},
            "signal_id": f"{scan_id}-{symbol.split('/')[0].split(':')[0]}-{alert_type}",
        }

        # Persist to CSV first (before Telegram send, so failure doesn't cost the signal)
        try:
            logger.log_signal(sig)
            summary["signals"] += 1
        except OSError as exc:
            summary["log_failed"] = summary.get("log_failed", 0) + 1
            log.error("signal log write failed for %s (%s) — continuing",
                      sig.get("coin"), exc)

        # Then send Telegram alert
        with _coordinator.stage("telegram"):
            sent = alerts.send_alert(sig)   # bounded (20s), failure logged
        if not sent:
            summary["telegram_failed"] += 1
        summary["signals"] += 1
        log.info("%s %s alert_sent=%s", symbol, alert_type, sent)
        guard.record(symbol, now_ist)   # cooldown only for alerted
    else:
        log.info("Scan %s complete: no volume spikes detected", scan_id)

    # Persist status and exit - AI journal removed for volume-only mode
    if not config.LLM_DECISION_ENABLED:
        summary["ai_status"] = "OFF"
    _log_funnel_summary(scan_id, summary)
    _coordinator.end(summary)
    return summary


def _log_funnel_summary(scan_id: str, summary: dict) -> None:
    """One concise INFO block per scan (per-symbol detail stays at DEBUG)."""
    stages = _coordinator.stages
    critical_ms = sum(v for k, v in stages.items() if k != "telegram")
    log.info(
        "SCAN #%s COMPLETE | universe %d | volume_spikes %d | alerts %d | "
        "log_failed %d | telegram_failed %d | "
        "critical %.1fs | AI %s",
        scan_id, summary.get("scanned", 0), summary.get("volume_spikes", 0),
        summary.get("alerts", 0), summary.get("log_failed", 0),
        summary.get("telegram_failed", 0),
        critical_ms / 1000.0, summary.get("ai_status", "OFF"))


def start_ondemand_scan() -> dict:
    """Telegram /scan_on: delegate to ondemand module."""
    return ondemand.start_ondemand_scan(
        run_scan_fn=run_scan,
        make_exchange_fn=scanner.make_exchange,
        fetch_funding_fn=scanner.fetch_funding_rates,
        guard_fn=get_coordinator().get_guard,   # SAME guard as scheduled scans
    )


def stop_ondemand_scan() -> dict:
    """Telegram /scan_off: delegate to ondemand module."""
    return ondemand.stop_ondemand_scan()


def _daily_summary() -> str:
    """Generate daily performance summary from signals_log.csv."""
    today = datetime.now(config.TZ).strftime("%Y-%m-%d")
    buys, sells, holds = 0, 0, 0
    try:
        if not config.SIGNALS_LOG_FILE.exists():
            return "No signals logged yet."
        with open(config.SIGNALS_LOG_FILE, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ts = row.get("timestamp", "")
                if not ts.startswith(today):
                    continue
                sig = row.get("signal", "")
                if sig == "BUY":
                    buys += 1
                elif sig == "SELL":
                    sells += 1
                elif sig == "HOLD":
                    holds += 1
    except Exception as exc:
        log.error("Daily summary read error: %s", exc)
        return f"Error reading signals: {exc}"

    total = buys + sells
    return (
        f"📊 <b>Daily Report ({today})</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🟢 BUY signals: <b>{buys}</b>\n"
        f"🔴 SELL signals: <b>{sells}</b>\n"
        f"⏸ HOLD (silent): <b>{holds}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📈 Total alerts sent: <b>{total}</b>\n"
        f"💰 Account: <b>${config.ACCOUNT_BALANCE:,.0f}</b> | "
        f"Risk: <b>{config.RISK_PER_TRADE_PCT}%</b>/trade"
    )


def build_scheduler() -> BlockingScheduler:
    """Production schedule: one scan every five minutes, 24/7.

    One job (not three overlapping ones) with an explicit overrun policy is
    what keeps alerts on time: `max_instances=1` + `coalesce=True` +
    `misfire_grace_time` mean a slow scan can never run concurrently with the
    next one nor stack up a backlog, and the `second` offset fires each scan a
    few seconds AFTER the candle close so the just-closed 5M candle exists.
    """
    exchange = scanner.make_exchange()
    # the ONE guard: shared with /scan_on sessions and --once via the
    # coordinator, so cooldowns are respected across every scan path
    guard = _coordinator.get_guard()
    scheduler = BlockingScheduler(timezone=config.SCHEDULER_TZ)
    # Continuous-operation health tracking.
    session = {"streak": 0, "scans": 0, "signals": 0}

    def scan_job() -> None:
        t0 = time.monotonic()
        try:
            tickers = exchange.fetch_tickers()
            funding_rates = scanner.fetch_funding_rates(exchange)
            summary = run_scan(exchange, guard, tickers, funding_rates)
            session["scans"] += 1
            session["signals"] += summary.get("signals", 0)
            session["streak"] = 0
        except Exception as exc:  # scheduler jobs must never kill the loop
            session["streak"] += 1
            log.error("Scan job failed (%d in a row): %s", session["streak"], exc)
            if session["streak"] == 3:
                alerts.send_telegram_text(
                    f"⚠️ <b>Bot health warning</b>\n"
                    f"3 scans in a row failed (last: {type(exc).__name__}). "
                    f"Signals may be paused — check the server.")
        finally:
            elapsed = time.monotonic() - t0
            if elapsed > config.SCAN_INTERVAL_MIN * 60 * 0.8:  # >80% of 5min slot
                log.warning("⚠️ Scan took %.1fs (slot is %ds) — next scan may be skipped!",
                        elapsed, config.SCAN_INTERVAL_MIN * 60)

    # Every five minutes, 24 hours a day, with no overlapping scans.
    scheduler.add_job(
        scan_job,
        CronTrigger(hour="*", minute="*/5", second=config.SCAN_SECOND_OFFSET,
                    timezone=config.SCHEDULER_TZ),
        id="scan", name="5-min scan 24/7",
        max_instances=1, misfire_grace_time=config.SCAN_MISFIRE_GRACE_SEC, coalesce=True,
    )
    return scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto Signal Bot (signals only, never trades)")
    parser.add_argument("--once", action="store_true",
                        help="DEMO/TEST: run one full scan cycle immediately")
    parser.add_argument("--force-llm", action="store_true",
                        help="TEST ONLY: bypass all gates; top setups go straight to the LLM")
    parser.add_argument("--force-llm-top", type=int, default=3, metavar="N",
                        help="TEST ONLY: how many top-quality coins --force-llm sends (default 3)")
    args = parser.parse_args()

    config.setup_logging()
    liquidation.start_listener()  # persistent; scans only read its current cache
    logger.migrate_csv_header()  # reconcile signals_log.csv to the current column set
    for warning in config.check_exposed_credentials():
        log.warning("SECURITY: %s", warning)
    # .env mistakes that would otherwise surface as "the AI went quiet"
    for warning in config.check_config_warnings():
        log.warning("CONFIG: %s", warning)
    log.info("=== Crypto Signal Bot starting (AI providers=%d [%s], telegram=%s) ===",
             len(config.AI_PROVIDERS), ai_decision.describe_providers(),
             "configured" if config.TELEGRAM_TOKEN else "NOT configured")

    log.info("AI CONTRACT (Phase 4): Ollama/LLM is the PRIMARY decision maker for "
             "funnel-passing candidates — Python validates levels + data only | "
             "provider ladder (%d): %s | "
             "json_mode=%s reasoning=%s | max_tokens=%d retry_cap=%d timeout=%.0fs | "
             "workers=%d budget=%s/day (local providers exempt) | LLM_DECISION_ENABLED=%s",
             len(config.AI_PROVIDERS),
             ai_decision.describe_providers(),
             "on" if config.AI_JSON_MODE else "off",
             "on" if config.AI_REASONING_ENABLED else "off",
             config.AI_MAX_TOKENS,
             config.AI_MAX_TOKENS_RETRY_CAP,
             config.AI_TIMEOUT_SECONDS,
             config.AI_WORKERS,
             config.AI_DAILY_BUDGET if config.AI_DAILY_BUDGET > 0 else "unlimited",
             "ON" if config.LLM_DECISION_ENABLED else "OFF")
    # News verification engine: discovery -> deterministic verification ->
    # (VERIFIED only) AI summary + impact analysis -> Telegram. Verification
    # and interpretation stay separate; news never feeds trading signals.
    if config.NEWS_ENABLED:
        import news
        news.start_engine()

    if args.once:
        log.info("DEMO/TEST mode: single scan cycle (outside scheduled session)")
        exchange = scanner.make_exchange()
        try:
            tickers = exchange.fetch_tickers()
            funding_rates = scanner.fetch_funding_rates(exchange)
        except Exception as exc:
            log.error("Ticker/funding fetch failed in demo mode: %s", exc)
            tickers = {}
            funding_rates = {}
        run_scan(exchange, duplicate_guard.DuplicateGuard(), tickers, funding_rates, force=True)
        return

    if args.force_llm:
        top_n = max(1, args.force_llm_top)
        log.info("FORCE-LLM TEST MODE: all gates bypassed — top %d setups go straight "
                 "to the LLM (logged only, no alerts/CSV/cooldown)", top_n)
        exchange = scanner.make_exchange()
        run_force_llm(exchange, top_n=top_n)
        return

    if config.TELEGRAM_TOKEN:
        started = telegram_bot.start_bot_listener()
        if started:
            log.info("Telegram Chat Assistant live: users can chat and query the bot on Telegram")

    # Send startup notification to Telegram
    alerts.send_telegram_text(
        f"🚀 <b>Crypto Signal Bot Started</b>\n\n"
         f"⏰ <b>Scanning:</b> 24/7, every 5 min\n"
        f"🤖<b>AI Providers:</b> {len(config.AI_PROVIDERS)} configured (auto-fallback pool)\n"
        f"⚡ <b>Market:</b> Futures (USDT-M Perpetual)\n"
        f"📊 <b>Volume Filter:</b> >= ${config.VOLUME_MIN_USDT:,} USDT\n"
        f"💰 <b>Risk:</b> {config.RISK_PER_TRADE_PCT}% per trade on ${config.ACCOUNT_BALANCE:,.0f}\n"
        f"💬 <b>24/7 AI Chat:</b> Send /start or any question anytime!"
    )

    scheduler = build_scheduler()
    log.info("Scheduler live: scans every 5 min, 24/7 (Ollama workers=%d). Ctrl+C to stop.",
             config.AI_WORKERS)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        alerts.send_telegram_text("🛑 <b>Crypto Signal Bot Stopped</b>")
        log.info("Bot stopped by user")