"""Shared test fixtures: synthetic OHLCV candle builders + a synthetic AI pool."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- hermetic config (set BEFORE config is imported anywhere in the suite) ---
# A deployment .env sitting in the repo root (server OLLAMA_BASE_URL,
# AI_MAX_TOKENS=800, AI_BATCH_MAX=1, ...) must not become config reality
# inside tests. config.py honours this flag by skipping load_dotenv(); the
# pop-list below additionally strips ambient exports from the operator's
# shell so the suite always sees factory defaults. Tests that need specific
# values set them explicitly via monkeypatch.
os.environ["ZCODE_HERMETIC_TESTS"] = "1"
for _k in (
    "OLLAMA_BASE_URL", "OLLAMA_MODEL", "AI_BASE_URL", "AI_MODEL",
    "AI_MODEL_FALLBACK", "OPENROUTER_API_KEY", "NVIDIA_API_KEY",
    "AGENTROUTER_API_KEY", "AI_DAILY_BUDGET", "AI_MAX_TOKENS",
    "AI_MAX_TOKENS_RETRY_CAP", "AI_BATCH_MAX", "AI_BATCH_ENABLED",
    "AI_JSON_MODE", "VOLUME_MIN_USDT", "MIN_RR",
):
    os.environ.pop(_k, None)

import config  # noqa: E402  (after sys.path so the repo copy is imported)


def make_candles(closes, start="2026-08-14 10:00", freq="1h", vol_base=1000.0,
                 vol_spread=0.1, wick=0.4, seed=7, volumes=None):
    """Build an OHLCV DataFrame from a close-price series.

    `volumes` overrides generated volume; `wick` scales the random high/low
    protrusion around the open/close body.
    """
    closes = np.asarray(closes, dtype="float64")
    n = len(closes)
    rng = np.random.default_rng(seed)
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    spread = np.maximum(np.abs(closes - opens), 1e-9)
    highs = np.maximum(opens, closes) + wick * spread * np.abs(rng.normal(1.0, 0.2, n))
    lows = np.minimum(opens, closes) - wick * spread * np.abs(rng.normal(1.0, 0.2, n))
    if volumes is None:
        vols = vol_base * (1.0 + vol_spread * rng.normal(0, 1, n))
        vols = np.abs(vols)
    else:
        vols = np.asarray(volumes, dtype="float64")
        assert len(vols) == n
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


@pytest.fixture
def candle_builder():
    return make_candles


@pytest.fixture(autouse=True)
def _ai_provider_pool(monkeypatch):
    """A synthetic provider pool when the environment has none.

    `config.AI_PROVIDERS` is built once at import from `.env`, so a test run
    without a configured key used to fail every transport test with
    "No AI providers configured" before the test's own mocks were even
    reached. Tests never touch the network (they mock `requests.post`); this
    only gives `_complete()` a pool to walk. A test can still opt out of AI
    entirely by patching `config.AI_PROVIDERS = []` itself.
    """
    if not config.AI_PROVIDERS:
        pool = [{"name": "test-primary",
                 "base_url": (config.AI_BASE_URL or "https://openrouter.ai/api/v1").rstrip("/"),
                 "api_key": "sk-or-test", "model": config.AI_MODEL or "test-model"}]
        monkeypatch.setattr(config, "AI_PROVIDERS", pool)
    yield


# ---- Phase 2 orchestration helpers -----------------------------------------
# Since strategy_core became the primary gate, decision-level tests assert the
# ORCHESTRATION contract (funnel -> single HTF veto -> risk gate -> verdict).
# The risk gate's internals have their own suite (test_risk_gate.py), so these
# helpers swap in a deterministic risk outcome instead of crafting candle
# geometry that satisfies the structural stop/target maths.

def make_risk_pass(monkeypatch):
    """risk_gate.evaluate -> ok with sane levels (SL 2% away, TP 6%, RR 3.0)."""
    import decision

    def fake(direction, entry, structure, sr, atr, cfg=config, spread_pct=None):
        long_ = direction == "LONG"
        sl = entry * (0.98 if long_ else 1.02)
        tp = entry * (1.06 if long_ else 0.94)
        return {"ok": True, "entry": round(entry, 8), "sl": sl, "tp": tp,
                "rr": 3.0, "risk": abs(entry - sl), "reward": abs(tp - entry),
                "target_zone": "test-zone", "target_skipped": 0,
                "target_note": "", "reasons": []}

    monkeypatch.setattr(decision.risk_gate, "evaluate", fake)


def make_risk_fail(monkeypatch, reasons=("poor_rr",)):
    """risk_gate.evaluate -> blocked with the given hard reasons."""
    import decision

    def fake(direction, entry, structure, sr, atr, cfg=config, spread_pct=None):
        return {"ok": False, "entry": entry, "sl": None, "tp": None, "rr": None,
                "risk": None, "reward": None, "target_zone": "",
                "target_skipped": 0, "target_note": "",
                "reasons": list(reasons)}

    monkeypatch.setattr(decision.risk_gate, "evaluate", fake)
