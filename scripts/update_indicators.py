#!/usr/bin/env python3
"""
Fetch all indicator data and write to indicators/data.json + indicators/history.csv.

Self-contained script for GitHub Actions (no dependency on parent repo).
API keys via environment variables.
"""

import csv
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_JSON = ROOT_DIR / "indicators" / "data.json"
HISTORY_CSV = ROOT_DIR / "indicators" / "history.csv"

# ============ Parameters ============

BTC_GENESIS = datetime(2009, 1, 3)
AHR999_ACTION = 0.45
FNG_LEVELS = {"重仓": 10, "行动": 15, "准备": 20, "关注": 25}
FUNDING_CONSECUTIVE = {"关注": 3, "准备": 5, "行动": 7}
NUPL_LEVELS = {"关注": 0.1, "行动": 0}  # Total NUPL (via BGeometrics)
MVRV_ZSCORE_LEVELS = {"关注": 0.5, "行动": 0}
PUELL_LEVELS = {"关注": 0.7, "行动": 0.5, "重仓": 0.3}
VIX_LEVELS = {"关注": 30, "准备": 35, "行动": 40, "重仓": 50}
MA200_LEVELS = {"关注": -10, "行动": -15, "重仓": -20}
SPY_GLD_THRESHOLDS = (7.0, 3.0)
SPY_TLT_THRESHOLDS = (7.0, 2.0)
DOUBLE_DROP_LOOKBACK = 7
VC_MONTHLY_THRESHOLD = 2500
VC_DEALS_THRESHOLD = 100
RATIO_MA_WINDOW = 30
RATIO_SLOPE_WINDOW = 5
RATIO_SURGE_THRESHOLD = 5.0
RATIO_REVERSAL_DAYS = 3


# ============ Helpers ============

def bgeometrics_get(endpoint):
    """Fetch data from BGeometrics free API (no key required). Returns list of dicts."""
    try:
        r = requests.get(f"https://bitcoin-data.com/api/v1/{endpoint}", timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  BGeometrics error ({endpoint}): {e}")
    return None


def download_prices(tickers, days=30):
    end = datetime.now()
    start = end - timedelta(days=days)
    data = yf.download(tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"] if "Close" in data.columns.get_level_values(0) else data["Adj Close"]
    else:
        prices = data["Close"] if "Close" in data.columns else data["Adj Close"]
    if "^VIX" in prices.columns:
        prices = prices.rename(columns={"^VIX": "VIX"})
    return prices


def calculate_return(prices, ticker, lookback):
    if ticker not in prices.columns:
        return None
    current = prices[ticker].iloc[-1]
    past = prices[ticker].iloc[-lookback - 1] if len(prices) > lookback else prices[ticker].iloc[0]
    return (current / past - 1) * 100


# ============ Indicator Fetchers ============

def fetch_ahr999():
    """Ahr999 index via CoinGecko."""
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": 200, "interval": "daily"}
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        closes = [p[1] for p in data["prices"]]
        if len(closes) < 200:
            return None
        current_price = closes[-1]
        sma_200 = sum(closes[-200:]) / 200
        days_since_genesis = (datetime.now() - BTC_GENESIS).days
        power_law_price = 10 ** (5.84 * math.log10(days_since_genesis) - 17.01)
        ahr999 = (current_price / sma_200) * (current_price / power_law_price)

        triggered = ahr999 < AHR999_ACTION
        if triggered:
            level = "行动"
            display = f"{ahr999:.2f} (低于阈值)"
        elif ahr999 < 1.2:
            level = "定投"
            display = f"{ahr999:.2f} (定投区)"
        else:
            level = None
            display = f"{ahr999:.2f}"

        return {
            "value": round(ahr999, 2),
            "display": display,
            "triggered": triggered,
            "level": level,
            "btc_price": round(current_price, 2),
        }
    except Exception as e:
        print(f"  Ahr999 error: {e}")
        return None


def fetch_nupl():
    """Total NUPL via BGeometrics (replaces STH-NUPL)."""
    try:
        data = bgeometrics_get("nupl")
        if not data:
            return None
        value = float(data[-1]["nupl"])
        triggered = value < NUPL_LEVELS["行动"]
        level = "行动" if triggered else ("关注" if value < NUPL_LEVELS["关注"] else None)
        return {
            "value": round(value, 3),
            "display": f"NUPL {value:.3f}",
            "triggered": triggered,
            "level": level,
        }
    except Exception as e:
        print(f"  NUPL error: {e}")
        return None


def fetch_mvrv_zscore():
    """MVRV Z-Score via BGeometrics."""
    try:
        data = bgeometrics_get("mvrv-zscore")
        if not data:
            return None
        value = float(data[-1]["mvrvZscore"])
        triggered = value < MVRV_ZSCORE_LEVELS["行动"]
        level = "行动" if triggered else ("关注" if value < MVRV_ZSCORE_LEVELS["关注"] else None)
        return {
            "value": round(value, 2),
            "display": f"Z-Score {value:.2f}",
            "triggered": triggered,
            "level": level,
        }
    except Exception as e:
        print(f"  MVRV Z-Score error: {e}")
        return None


def fetch_puell_multiple():
    """Puell Multiple via BGeometrics."""
    try:
        data = bgeometrics_get("puell-multiple")
        if not data:
            return None
        value = float(data[-1]["puellMultiple"])
        if value < PUELL_LEVELS["重仓"]:
            level = "重仓"
        elif value < PUELL_LEVELS["行动"]:
            level = "行动"
        elif value < PUELL_LEVELS["关注"]:
            level = "关注"
        else:
            level = None
        triggered = level in ("行动", "重仓")
        return {
            "value": round(value, 2),
            "display": f"Puell {value:.2f}",
            "triggered": triggered,
            "level": level,
        }
    except Exception as e:
        print(f"  Puell error: {e}")
        return None


def fetch_hash_ribbons():
    """Hash Ribbons via BGeometrics hash rate (MA30/MA60 crossover)."""
    try:
        data = bgeometrics_get("hashrate")
        if not data or len(data) < 70:
            return None
        # Use last 90 entries
        recent = data[-90:]
        values = [float(d["hashrate"]) for d in recent]
        dates = [d["d"] for d in recent]
        df = pd.DataFrame({"hash_rate": values}, index=pd.to_datetime(dates))
        df["ma30"] = df["hash_rate"].rolling(30).mean()
        df["ma60"] = df["hash_rate"].rolling(60).mean()
        df = df.dropna()
        if len(df) < 2:
            return None

        current_ma30 = df["ma30"].iloc[-1]
        current_ma60 = df["ma60"].iloc[-1]
        prev_ma30 = df["ma30"].iloc[-2]
        prev_ma60 = df["ma60"].iloc[-2]
        capitulating = current_ma30 < current_ma60
        crossover = (prev_ma30 < prev_ma60) and (current_ma30 >= current_ma60)
        ratio = (current_ma30 / current_ma60 - 1) * 100

        if crossover:
            state = "投降结束"
            level = "行动"
        elif capitulating:
            state = "投降中"
            level = "投降"
        else:
            state = "正常"
            level = None

        if crossover:
            numeric = 1
        elif capitulating:
            numeric = -1
        else:
            numeric = 0

        return {
            "value": numeric,
            "display": f"{state} (MA30/MA60: {ratio:+.1f}%)",
            "triggered": crossover,
            "level": level,
        }
    except Exception as e:
        print(f"  Hash Ribbons error: {e}")
        return None


def fetch_fear_greed():
    """Crypto Fear & Greed Index."""
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=10)
        data = r.json()["data"][0]
        value = int(data["value"])
        label = data["value_classification"]

        if value < FNG_LEVELS["重仓"]:
            level = "重仓"
        elif value < FNG_LEVELS["行动"]:
            level = "行动"
        elif value < FNG_LEVELS["准备"]:
            level = "准备"
        elif value < FNG_LEVELS["关注"]:
            level = "关注"
        else:
            level = None

        triggered = level in ("行动", "重仓")
        return {
            "value": value,
            "display": f"{value} ({label})",
            "triggered": triggered,
            "level": level,
        }
    except Exception as e:
        print(f"  Fear & Greed error: {e}")
        return None


def fetch_funding_rate():
    """BTC perpetual funding rate from Bybit (no US geo-block)."""
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/funding/history",
            params={"category": "linear", "symbol": "BTCUSDT", "limit": 21},
            timeout=10,
        )
        data = r.json()
        if data.get("retCode") != 0:
            return None
        items = data["result"]["list"]
        if not items:
            return None
        # Bybit returns newest first
        rates = [float(d["fundingRate"]) * 100 for d in reversed(items)]
        current_rate = rates[-1]
        neg_streak = 0
        for rate in reversed(rates):
            if rate < 0:
                neg_streak += 1
            else:
                break

        for name, threshold in sorted(FUNDING_CONSECUTIVE.items(), key=lambda x: x[1], reverse=True):
            if neg_streak >= threshold:
                level = name
                break
        else:
            level = None

        triggered = level in ("准备", "行动")
        return {
            "value": neg_streak,
            "display": f"当前 {current_rate:.4f}%，连续负 {neg_streak} 次",
            "triggered": triggered,
            "level": level,
            "current_rate": round(current_rate, 4),
        }
    except Exception as e:
        print(f"  Funding Rate error: {e}")
        return None


def fetch_stablecoin_supply():
    """Stablecoin supply growth from DeFiLlama."""
    try:
        r = requests.get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=15)
        data = r.json()
        if len(data) < 91:
            return None
        latest = data[-1]["totalCirculatingUSD"]["peggedUSD"]
        past_90d = data[-91]["totalCirculatingUSD"]["peggedUSD"]
        growth_90d = (latest / past_90d - 1) * 100
        latest_b = latest / 1e9
        level = "收缩" if growth_90d < 0 else None
        return {
            "value": round(growth_90d, 1),
            "display": f"${latest_b:.1f}B, 90d {growth_90d:+.1f}%",
            "triggered": False,
            "level": level,
        }
    except Exception as e:
        print(f"  Stablecoin error: {e}")
        return None


def fetch_vc_funding():
    """Crypto VC funding from DeFiLlama."""
    try:
        r = requests.get("https://api.llama.fi/raises", timeout=15)
        data = r.json()
        raises = data.get("raises", [])
        if not raises:
            return None

        now_ts = datetime.now().timestamp()
        ts_90d = now_ts - 90 * 86400
        ts_30d = now_ts - 30 * 86400

        recent_3m = [r for r in raises if r.get("date") and r["date"] >= ts_90d]
        total_3m = sum(r.get("amount") or 0 for r in recent_3m)
        avg_monthly_3m = total_3m / 3

        recent_1m = [r for r in raises if r.get("date") and r["date"] >= ts_30d]
        deals_1m = len(recent_1m)

        triggered = avg_monthly_3m > VC_MONTHLY_THRESHOLD or deals_1m > VC_DEALS_THRESHOLD
        level = "过热" if triggered else None

        return {
            "value": round(avg_monthly_3m, 0),
            "display": f"${avg_monthly_3m:,.0f}M/月 (3M均), {deals_1m} deals/月",
            "triggered": triggered,
            "level": level,
            "deals_1m": deals_1m,
        }
    except Exception as e:
        print(f"  VC Funding error: {e}")
        return None


def fetch_tradfi_indicators():
    """Fetch VIX, MA200 deviation, double drops, SCHD/QQQ ratio."""
    results = {}

    # Download market data
    print("  Downloading market data...")
    prices = download_prices(["SPY", "GLD", "TLT", "^VIX"], days=30)

    # -- VIX --
    try:
        if "VIX" in prices.columns:
            vix = float(prices["VIX"].iloc[-1])
            level = None
            for name, threshold in sorted(VIX_LEVELS.items(), key=lambda x: x[1], reverse=True):
                if vix > threshold:
                    level = name
                    break
            triggered = level in ("行动", "重仓")
            results["vix-spike"] = {
                "value": round(vix, 1),
                "display": f"VIX {vix:.1f}",
                "triggered": triggered,
                "level": level,
            }
    except Exception as e:
        print(f"  VIX error: {e}")

    # -- MA200 Deviation --
    try:
        spy_2y = yf.download("SPY", period="2y", progress=False)
        if isinstance(spy_2y.columns, pd.MultiIndex):
            close = spy_2y["Close"]["SPY"]
        else:
            close = spy_2y["Close"]
        ma200 = close.rolling(200).mean()
        current_price = float(close.iloc[-1])
        current_ma200 = float(ma200.iloc[-1])
        deviation = (current_price / current_ma200 - 1) * 100

        level = None
        for name, threshold in sorted(MA200_LEVELS.items(), key=lambda x: x[1]):
            if deviation < threshold:
                level = name
        triggered = level in ("行动", "重仓")
        results["ma200-deviation"] = {
            "value": round(deviation, 1),
            "display": f"偏离 {deviation:+.1f}%",
            "triggered": triggered,
            "level": level,
        }
    except Exception as e:
        print(f"  MA200 error: {e}")

    # -- Double Drops --
    try:
        spy_ret = calculate_return(prices, "SPY", DOUBLE_DROP_LOOKBACK)
        gld_ret = calculate_return(prices, "GLD", DOUBLE_DROP_LOOKBACK)
        tlt_ret = calculate_return(prices, "TLT", DOUBLE_DROP_LOOKBACK)

        # SPY+GLD
        triggered_gld = False
        if spy_ret and gld_ret:
            triggered_gld = spy_ret < -SPY_GLD_THRESHOLDS[0] and gld_ret < -SPY_GLD_THRESHOLDS[1]
        results["double-drop-spy-gld"] = {
            "value": f"{spy_ret:.2f},{gld_ret:.2f}" if spy_ret and gld_ret else "",
            "display": f"SPY {spy_ret:.2f}%, GLD {gld_ret:.2f}%" if spy_ret and gld_ret else "N/A",
            "triggered": triggered_gld,
            "level": None,
        }

        # SPY+TLT
        triggered_tlt = False
        if spy_ret and tlt_ret:
            triggered_tlt = spy_ret < -SPY_TLT_THRESHOLDS[0] and tlt_ret < -SPY_TLT_THRESHOLDS[1]
        results["double-drop-spy-tlt"] = {
            "value": f"{spy_ret:.2f},{tlt_ret:.2f}" if spy_ret and tlt_ret else "",
            "display": f"SPY {spy_ret:.2f}%, TLT {tlt_ret:.2f}%" if spy_ret and tlt_ret else "N/A",
            "triggered": triggered_tlt,
            "level": None,
        }
    except Exception as e:
        print(f"  Double drop error: {e}")

    # -- SCHD/QQQ Ratio --
    try:
        end = datetime.now()
        start = end - timedelta(days=180)
        schd_qqq = yf.download(["SCHD", "QQQ"], start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"), progress=False)
        if isinstance(schd_qqq.columns, pd.MultiIndex):
            rp = schd_qqq["Close"] if "Close" in schd_qqq.columns.get_level_values(0) else schd_qqq["Adj Close"]
        else:
            rp = schd_qqq
        rp = rp.dropna()

        if len(rp) >= RATIO_MA_WINDOW + 10:
            schd_norm = rp["SCHD"] / rp["SCHD"].iloc[0] * 100
            qqq_norm = rp["QQQ"] / rp["QQQ"].iloc[0] * 100
            ratio = schd_norm / qqq_norm * 100
            ratio_ma = ratio.rolling(window=RATIO_MA_WINDOW).mean()

            current_ratio = float(ratio.iloc[-1])
            ma_slope = float(ratio_ma.iloc[-1] - ratio_ma.iloc[-RATIO_SLOPE_WINDOW - 1])

            # Check for reversal
            ma_diffs = ratio_ma.diff()
            neg_streak = 0
            for v in reversed(ma_diffs.dropna().values):
                if v < 0:
                    neg_streak += 1
                else:
                    break
            pos_streak = 0
            for v in reversed(ma_diffs.dropna().values):
                if v > 0:
                    pos_streak += 1
                else:
                    break

            if len(ratio) > 60:
                ratio_60d_change = float((ratio.iloc[-1] / ratio.iloc[-61] - 1) * 100)
            else:
                ratio_60d_change = float((ratio.iloc[-1] / ratio.iloc[0] - 1) * 100)

            recent_60 = ratio.iloc[-min(60, len(ratio)):]
            ratio_peak = float(recent_60.max())
            ratio_from_peak = (current_ratio / ratio_peak - 1) * 100

            trend = "↑ SCHD 走强 (risk-off)" if ma_slope > 0 else "↓ QQQ 走强 (risk-on)"

            # Signal logic
            level = None
            triggered = False
            if pos_streak >= 10 and ratio_60d_change > 3:
                level = "防御"
            had_surge = ratio_60d_change > RATIO_SURGE_THRESHOLD or ratio_from_peak < -2
            slope_reversed = neg_streak >= RATIO_REVERSAL_DAYS
            if had_surge and slope_reversed and ratio_from_peak < -1:
                level = "抄底"
                triggered = True

            results["schd-qqq-ratio"] = {
                "value": round(current_ratio, 2),
                "display": f"比率 {current_ratio:.2f} ({trend})",
                "triggered": triggered,
                "level": level,
            }
    except Exception as e:
        print(f"  SCHD/QQQ error: {e}")

    return results


# ============ Main ============

def collect_all():
    """Collect all indicator data."""
    indicators = {}

    print("Fetching Crypto indicators...")
    for name, func in [
        ("ahr999", fetch_ahr999),
        ("nupl", fetch_nupl),
        ("mvrv-zscore", fetch_mvrv_zscore),
        ("puell-multiple", fetch_puell_multiple),
        ("hash-ribbons", fetch_hash_ribbons),
        ("fear-greed-crypto", fetch_fear_greed),
        ("funding-rate", fetch_funding_rate),
        ("stablecoin-supply", fetch_stablecoin_supply),
        ("crypto-vc-funding", fetch_vc_funding),
    ]:
        print(f"  {name}...")
        result = func()
        if result:
            indicators[name] = result
        else:
            indicators[name] = {"value": None, "display": "获取失败", "triggered": False, "level": None}

    print("Fetching TradFi indicators...")
    tradfi = fetch_tradfi_indicators()
    indicators.update(tradfi)

    # Manual indicators (no auto data)
    for key in ("miner-shutdown-price", "etf-net-directional-flow", "ico-market-heat"):
        if key not in indicators:
            indicators[key] = {"value": None, "display": None, "triggered": False, "level": None}

    return indicators


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_data_json(indicators):
    """Write indicators/data.json."""
    DATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "indicators": indicators,
    }
    with open(DATA_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        f.write("\n")
    print(f"Written {DATA_JSON}")


def append_history_csv(indicators):
    """Append a row to indicators/history.csv (one row per day, dedup)."""
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Extract numeric values
    def get_val(key, field="value"):
        ind = indicators.get(key, {})
        v = ind.get(field)
        if v is None:
            return ""
        return v

    btc_price = indicators.get("ahr999", {}).get("btc_price", "")

    row = {
        "date": today,
        "ahr999": get_val("ahr999"),
        "nupl": get_val("nupl"),
        "mvrv_zscore": get_val("mvrv-zscore"),
        "puell_multiple": get_val("puell-multiple"),
        "hash_ribbons": get_val("hash-ribbons"),
        "fear_greed": get_val("fear-greed-crypto"),
        "funding_neg_streak": get_val("funding-rate"),
        "stablecoin_growth_90d": get_val("stablecoin-supply"),
        "vc_funding_monthly": get_val("crypto-vc-funding"),
        "vix": get_val("vix-spike"),
        "spy_ma200_dev": get_val("ma200-deviation"),
        "btc_price": btc_price,
    }

    fields = list(row.keys())

    # Read existing, dedup today
    existing_rows = []
    if HISTORY_CSV.exists():
        with open(HISTORY_CSV, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("date") != today:
                    existing_rows.append(r)

    # Write back
    with open(HISTORY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in existing_rows:
            # Only write fields that exist in the header
            writer.writerow({k: r.get(k, "") for k in fields})
        writer.writerow(row)

    print(f"Appended history row for {today} to {HISTORY_CSV}")


def main():
    indicators = collect_all()
    write_data_json(indicators)
    append_history_csv(indicators)

    # Summary
    triggered = [k for k, v in indicators.items() if v.get("triggered")]
    if triggered:
        print(f"\nTriggered: {', '.join(triggered)}")
    else:
        print("\nNo indicators triggered.")


if __name__ == "__main__":
    main()
