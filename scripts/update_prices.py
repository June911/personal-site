#!/usr/bin/env python3
"""Fetch BTC/ETH/CRCL prices and write to prices.json."""

import json
import os
from datetime import datetime, timezone

import requests
import yfinance as yf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(SCRIPT_DIR, "..", "prices.json")


def fetch_crypto():
    """Fetch BTC and ETH prices from CoinGecko free API."""
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin,ethereum",
        "vs_currencies": "usd",
        "include_24hr_change": "true",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return {
        "BTC": {
            "price": round(data["bitcoin"]["usd"], 2),
            "change_24h": round(data["bitcoin"]["usd_24h_change"], 2),
        },
        "ETH": {
            "price": round(data["ethereum"]["usd"], 2),
            "change_24h": round(data["ethereum"]["usd_24h_change"], 2),
        },
    }


def fetch_stock():
    """Fetch CRCL price from yfinance."""
    ticker = yf.Ticker("CRCL")
    hist = ticker.history(period="2d")
    if len(hist) < 1:
        raise ValueError("No CRCL price data returned")
    current = hist["Close"].iloc[-1]
    if len(hist) >= 2:
        prev = hist["Close"].iloc[-2]
        change = ((current - prev) / prev) * 100
    else:
        change = 0.0
    return {
        "CRCL": {
            "price": round(float(current), 2),
            "change_24h": round(float(change), 2),
        }
    }


def main():
    result = {"updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

    try:
        result.update(fetch_crypto())
        print(f"BTC: ${result['BTC']['price']:,.2f} ({result['BTC']['change_24h']:+.2f}%)")
        print(f"ETH: ${result['ETH']['price']:,.2f} ({result['ETH']['change_24h']:+.2f}%)")
    except Exception as e:
        print(f"Crypto fetch failed: {e}")
        result.setdefault("BTC", {"price": 0, "change_24h": 0})
        result.setdefault("ETH", {"price": 0, "change_24h": 0})

    try:
        result.update(fetch_stock())
        print(f"CRCL: ${result['CRCL']['price']:.2f} ({result['CRCL']['change_24h']:+.2f}%)")
    except Exception as e:
        print(f"Stock fetch failed: {e}")
        result.setdefault("CRCL", {"price": 0, "change_24h": 0})

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
