"""Fear & Greed Index (FNG) downloader a týdenní agregátor.

- Stahuje denní FNG z alternative.me API
- Ukládá do CSV: data/fng_daily.csv (UTC data)
- Poskytuje týdenní agregaci zarovnanou k pondělí s více politikami zarovnání

DŮLEŽITÉ (časování):
- FNG: aktualizace ~ 00:00 UTC každého dne (time_until_update v API to potvrzuje)
- CMC historical YYYMMDD: end-of-day snapshot daného dne (EOD, ~ 23:59 UTC)

Politiky zarovnání v daily_to_weekly(policy):
- 'monday'  → pondělní FNG k pondělnímu EOD snapshotu (bez lookaheadu, nejaktuálnější platná informace)
- 'sunday'  → nedělní FNG k pondělnímu EOD (záměrně konzervativní, ~24h lag)
- 'friday'  → páteční FNG k pondělnímu EOD (extra konzervativní)
- 'tuesday_lookahead' → ÚTERNÍ FNG k pondělnímu EOD (ÚMYSLNÝ LOOKAHEAD) – používá se pouze na výslovný pokyn

POZOR: 'tuesday_lookahead' porušuje kauzalitu backtestu (lookahead bias) – výsledky mohou být
optimističtější než dosažitelné v reálu. Používejte pouze, pokud si to vědomě přejete.

Použití CLI:
    python -m indicators.fng --update [--out data/fng_daily.csv]

Programově:
    from indicators.fng import load_fng_daily_csv, daily_to_weekly
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal
from urllib.request import urlopen, Request

import pandas as pd

API_URL = "https://api.alternative.me/fng/?limit=0&format=json"
DEFAULT_OUTPUT = Path("data/fng_daily.csv")


def _fetch_fng_json() -> dict:
    req = Request(API_URL, headers={"User-Agent": "btc-dom-index/1.0"})
    with urlopen(req, timeout=60) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def fetch_fng_daily() -> pd.DataFrame:
    """Fetch daily FNG data and return a DataFrame with UTC dates.

    Columns:
        - date (Timestamp, normalized to 00:00, tz-naive UTC)
        - value (int)
        - value_classification (str)
    """
    raw = _fetch_fng_json()
    rows = raw.get("data", [])
    if not rows:
        return pd.DataFrame(columns=["date", "value", "value_classification"])  # empty

    df = pd.DataFrame(rows)
    # Convert timestamp (seconds) to UTC date (tz-naive)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    # Normalize columns
    df["value"] = df["value"].astype(int)
    df.rename(columns={"value_classification": "classification"}, inplace=True)
    return df[["date", "value", "classification"]].drop_duplicates().sort_values("date").reset_index(drop=True)


def save_fng_daily_csv(df: pd.DataFrame, path: Path | str = DEFAULT_OUTPUT) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_fng_daily_csv(path: Path | str = DEFAULT_OUTPUT) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        # If not present, try to fetch fresh
        df = fetch_fng_daily()
        save_fng_daily_csv(df, path)
        return df
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])  # tz-naive normalized dates
    # Ensure sorting and de-duplication
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def daily_to_weekly(df_daily: pd.DataFrame, policy: Literal["monday", "sunday", "friday", "tuesday_lookahead"] = "monday") -> pd.DataFrame:
    """Agregace denního FNG na týdenní hodnoty zarovnané k pondělkům.

    Politiky:
        - 'monday': použij pondělní hodnotu (bez lookaheadu – hodnota je známa od 00:00 UTC)
        - 'sunday': použij neděli (striktně trailing, ~24h lag vůči pondělnímu EOD)
        - 'friday': použij pátek (extra konzervativní)
        - 'tuesday_lookahead': použij ÚTERÝ pro pondělí (ÚMYSLNÝ LOOKAHEAD; porušuje kauzalitu)

    Návratová tabulka:
        rebalance_ts (Timestamp, pondělí 00:00)
        fng_value (int)
        fng_classification (str)
    """
    if df_daily.empty:
        return pd.DataFrame(columns=["rebalance_ts", "fng_value", "fng_classification"])  # empty

    s = df_daily.copy()
    s = s.set_index("date").asfreq("D").ffill()

    # Posun série vůči pondělnímu výběru
    if policy == "monday":
        shift_days = 0
    elif policy == "sunday":
        shift_days = 1
    elif policy == "friday":
        shift_days = 3
    elif policy == "tuesday_lookahead":
        shift_days = -1  # pondělí převezme úterní hodnotu (LOOKAHEAD)
    else:
        raise ValueError(f"Unsupported policy: {policy}")

    # UPOZORNĚNÍ na lookahead
    if policy == "tuesday_lookahead":
        print("\n[WARNING] FNG policy 'tuesday_lookahead' je zapnuta.")
        print("→ K pondělí se přiřazuje úterní hodnota FNG. Jedná se o ÚMYSLNÝ LOOKAHEAD.")
        print("→ Může vzniknout drobný časový rozdíl v řádu vteřin (technický LOOKAHEAD).\n")

    # Obecný posun (kladný = trailing, záporný = lookahead)
    s_shifted = s.shift(shift_days) if shift_days != 0 else s

    mondays = s_shifted[s_shifted.index.weekday == 0].copy()

    out = mondays.reset_index().rename(columns={"date": "rebalance_ts"})
    out = out.rename(columns={"value": "fng_value", "classification": "fng_classification"})
    # Jen potřebné sloupce
    out = out[["rebalance_ts", "fng_value", "fng_classification"]]
    return out


def _cli_update(output: Path | str = DEFAULT_OUTPUT) -> None:
    df = fetch_fng_daily()
    save_fng_daily_csv(df, output)
    print(f"Saved {len(df)} FNG daily rows to {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update Fear & Greed daily CSV")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV path")
    args = parser.parse_args()

    _cli_update(Path(args.out))

