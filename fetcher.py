"""Fetch historical weekly top tokens from CoinMarketCap with dynamic loading."""
# pip install playwright bs4 pandas tqdm
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import datetime as dt
import pathlib
import json
from tqdm import tqdm

## weekly snapshots on Mondays between 2021-05-09 and 2025-05-11 (START is inclusive lower bound)
START  = dt.date(2021, 5,  9)
END    = dt.date(2025, 5, 11)

# e.g. 100 to take top-100 rows per snapshot; None for no limit (all rows)
MAX_TOKENS = 100

# ensure snapshots folder exists
pathlib.Path("snapshots").mkdir(exist_ok=True)

def mondays(start, end):
    """Yield consecutive Mondays between start and end as YYYYMMDD strings."""
    # find first Monday on or after start
    shift = (0 - start.weekday() + 7) % 7
    d = start + dt.timedelta(days=shift)
    while d <= end:
        yield d.strftime("%Y%m%d")
        d += dt.timedelta(days=7)

def scrape_historical():
    """Scrape CoinMarketCap historical snapshots dynamically via Playwright."""
    weekly_frames = []

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()

        for date in tqdm(list(mondays(START, END))):
            url = f"https://coinmarketcap.com/historical/{date}/"
            page.goto(url, timeout=60000)
            # wait for table rows to appear
            page.wait_for_selector("tr.cmc-table-row")
            # incrementally scroll to load up to MAX_TOKENS rows via virtualization
            if MAX_TOKENS:
                # count loaded rows by symbol-cell presence
                loaded = page.evaluate(
                    "() => document.querySelectorAll('td.cmc-table__cell--sort-by__symbol').length")
                total_height = page.evaluate("() => document.body.scrollHeight")
                viewport = page.evaluate("() => window.innerHeight")
                step = max(200, int(viewport * 0.8))
                scroll_pos = 0
                # scroll until we have enough loaded symbols or reach bottom
                while loaded < MAX_TOKENS and scroll_pos < total_height:
                    scroll_pos += step
                    page.evaluate(f"() => window.scrollTo(0, {scroll_pos})")
                    page.wait_for_timeout(500)
                    loaded = page.evaluate(
                        "() => document.querySelectorAll('td.cmc-table__cell--sort-by__symbol').length")
            # final scroll to bottom to ensure all in-view
            page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(500)
            html = page.content()
            soup = BeautifulSoup(html, "html.parser")
            rows = soup.select("tr.cmc-table-row")
            if MAX_TOKENS:
                rows = rows[:MAX_TOKENS]
            for r in rows:
                cols = [c.get_text(strip=True) for c in r.select("td")]
                weekly_frames.append([date] + cols[:7])

        browser.close()

    # build DataFrame
    cols = ["snapshot_date","rank","name","symbol","market_cap","price","circulating","volume_24h"]
    df = pd.DataFrame(weekly_frames, columns=cols)
    # convert snapshot_date to datetime
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'], format='%Y%m%d')
    return df

if __name__ == '__main__':
    df = scrape_historical()
    df.to_csv("top100_weekly_2021-2025.csv", index=False)
