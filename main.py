# cmc_weekly_top50.py  ▶  pip install requests pandas python-dateutil lxml
import time, re, requests, pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta, MO, SU

BASE = "https://coinmarketcap.com/historical/{}"   # YYYYMMDD
OUT  = "cmc_top50_last_month.csv"
EX_STABLES = {"USDT", "USDC", "DAI", "TUSD", "FDUSD"}   # optional
WEEKS = 4               # ← změňte na 208 pro 4 roky

def completed_mondays(n_weeks=WEEKS):
    """Vrať n dokončených pondělků zpětně (dnes se nepočítá)."""
    today = datetime.now(timezone.utc).date()
    last_mon = today - timedelta(days=today.weekday() or 7)  # poslední HOTOVÝ po
    for _ in range(n_weeks):
        yield last_mon
        last_mon -= timedelta(weeks=1)

def monday_to_snapshot(monday):
    """CMC snapshot je nedělní => pondělí –1 den."""
    sunday = monday - timedelta(days=1)
    return sunday.strftime("%Y%m%d"), sunday

rows = []
for mon in completed_mondays():
    ymd, snap_date = monday_to_snapshot(mon)
    url = BASE.format(ymd)
    print(f"⬇  {snap_date}  – {url}")
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text

    # vezmi první tabulku, která obsahuje sloupec 'Market Cap'
    tables = pd.read_html(html, flavor="lxml", decimal='.', thousands=',')
    table = next(t for t in tables if any("Market Cap" in str(c) for c in t.columns))

    # sjednotíme názvy, odfiltrujeme stablecoiny a vezmeme TOP-50
    table.columns = [re.sub(r'[\s/]+', '_', str(c)).lower() for c in table.columns]
    top50 = (table
             .rename(columns={"market_cap": "market_cap_usd", "price": "price_usd"})
             .loc[~table['symbol'].isin(EX_STABLES)]
             .head(50)[["symbol", "price_usd", "market_cap_usd"]])
    top50["snapshot_date"] = mon        # logujeme pondělí 00:00 UTC
    rows.append(top50)
    time.sleep(1)                       # 1 req/s je slušnost

pd.concat(rows).to_csv(OUT, index=False)
print(f"✅  Uloženo → {OUT}")
