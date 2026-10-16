"""Phase 9 — Duplicate guard (24/7-correct).

Two independent cooldowns, both in-memory and thread-safe for the scheduler:

  * **Alert cooldown** (`DUPLICATE_COOLDOWN_MIN`, 20 min): a coin that got a
    BUY/SELL alert is not alerted again until it expires, and it is skipped
    *before* any 15M/5M fetch is paid for. Checked for every candidate,
    recorded only for alerted signals — a silent HOLD must not be able to mute
    a real setup that appears 5 minutes later.
  * **HOLD log cooldown** (`HOLD_LOG_COOLDOWN_MIN`, 30 min): the decision core
    rejects most coins on most scans, and every rejection used to append a
    fresh CSV row every 5 minutes (~2.2k HOLD rows per session, 24/7 in an
    on-demand session). An *identical* rejection — same verdict, same blocking
    reason — is written once per window; a new reason or a changed verdict is
    always written, because that is the information the owner reads.

**24/7 operation — no session reset.** The bot scans around the clock, so the
old "whole tracker resets at 23:00 IST" concept is gone (the reset job itself
had been removed from main.py, leaving `GUARD_RESET_TIME` dead config and
unbounded dict growth). Both cooldowns are timestamp-expiry based: an entry
simply stops mattering once its window has passed. `prune(now)` drops those
expired entries and is called once per scan, so memory stays bounded no matter
how many days the bot runs. `reset()` is kept for tests and explicit restarts.
"""
import logging
import threading
from datetime import datetime, timedelta

import config

log = logging.getLogger("duplicate_guard")


class DuplicateGuard:
    def __init__(self, cooldown_minutes: int = config.DUPLICATE_COOLDOWN_MIN,
                 hold_log_minutes: int = config.HOLD_LOG_COOLDOWN_MIN):
        self._cooldown = timedelta(minutes=cooldown_minutes)
        self._hold_cooldown = timedelta(minutes=hold_log_minutes)
        self._last_signal: dict[str, datetime] = {}
        self._last_hold: dict[str, tuple[datetime, str]] = {}
        self._lock = threading.Lock()

    def is_duplicate(self, symbol: str, now: datetime) -> bool:
        """True when this coin alerted within the alert cooldown window."""
        with self._lock:
            last = self._last_signal.get(symbol)
            return last is not None and (now - last) < self._cooldown

    def record(self, symbol: str, now: datetime) -> None:
        with self._lock:
            self._last_signal[symbol] = now

    def hold_logged_recently(self, symbol: str, now: datetime,
                             reason: str = "") -> bool:
        """True when this exact rejection was already written inside the window.

        `reason` is the verdict + blocking-reason signature; a different reason
        is never suppressed, so the log still shows a setup changing shape.
        """
        with self._lock:
            entry = self._last_hold.get(symbol)
        if entry is None:
            return False
        when, stored = entry
        return (now - when) < self._hold_cooldown and (not reason or stored == reason)

    def record_hold(self, symbol: str, now: datetime, reason: str = "") -> None:
        """Stamp a written HOLD row so the same blocked setup is not re-logged."""
        with self._lock:
            self._last_hold[symbol] = (now, reason)

    def prune(self, now: datetime) -> tuple[int, int]:
        """Drop entries whose cooldown window has expired (24/7 housekeeping).

        Cooldown correctness never depended on this — an expired entry compares
        False in `is_duplicate` / `hold_logged_recently` — but without pruning
        the dicts grow once per alerted/hold coin per day, forever. Called once
        per scan. Returns (alerts_pruned, holds_pruned).
        """
        with self._lock:
            stale_alerts = [s for s, t in self._last_signal.items()
                            if (now - t) >= self._cooldown]
            for s in stale_alerts:
                del self._last_signal[s]
            stale_holds = [s for s, (t, _) in self._last_hold.items()
                           if (now - t) >= self._hold_cooldown]
            for s in stale_holds:
                del self._last_hold[s]
        if stale_alerts or stale_holds:
            log.debug("Duplicate tracker pruned %d alert / %d hold entries",
                      len(stale_alerts), len(stale_holds))
        return len(stale_alerts), len(stale_holds)

    def reset(self) -> None:
        """Explicit full clear (tests / operator request) — NOT part of 24/7 ops."""
        with self._lock:
            alerts_n = len(self._last_signal)
            holds_n = len(self._last_hold)
            self._last_signal.clear()
            self._last_hold.clear()
        log.info("Duplicate tracker reset (%d alert / %d hold entries cleared)",
                 alerts_n, holds_n)

    def tracked_count(self) -> int:
        with self._lock:
            return len(self._last_signal)

    def hold_tracked_count(self) -> int:
        with self._lock:
            return len(self._last_hold)
