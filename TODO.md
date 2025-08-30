# TODOs

- Backfill missing historical snapshot for CoinMarketCap weekly data:
  - Missing week: 2017-07-10 (Monday)
  - Context: `https://coinmarketcap.com/historical/20170710/` currently fails to render the table (Playwright times out waiting for rows). Neighboring weeks load fine.
  - Plan:
    - Retry later with longer timeouts and alternative selectors for legacy markup.
    - Try different browser engines (Chromium/WebKit) and re-run.
    - If CMC restores the page, re-run `python fetcher.py --add-historical-from 2017-07-10` to fill the gap.
  - Status: Known issue; UI warns about the missing week.

