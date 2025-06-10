"""Fetch historical weekly top tokens from CoinMarketCap with dynamic loading."""
# pip install playwright bs4 pandas tqdm
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import datetime as dt
import pathlib
import json
import os
from tqdm import tqdm

## weekly snapshots on Mondays between 2021-05-09 and current date
START  = dt.date(2021, 5,  9)
END    = dt.date.today()

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

def get_last_date_from_csv(csv_file):
    """Get the last date from existing CSV file."""
    if not os.path.exists(csv_file):
        return None
    
    df = pd.read_csv(csv_file)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    return df['snapshot_date'].max().date()

def get_first_date_from_csv(csv_file):
    """Get the first date from existing CSV file."""
    if not os.path.exists(csv_file):
        return None
    
    df = pd.read_csv(csv_file)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    return df['snapshot_date'].min().date()

def scrape_historical(start_date=None):
    """Scrape CoinMarketCap historical snapshots dynamically via Playwright."""
    weekly_frames = []
    
    # If start_date is provided, use it; otherwise use global START
    actual_start = start_date if start_date else START

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()

        monday_list = list(mondays(actual_start, END))
        print(f"Scraping {len(monday_list)} snapshots from {actual_start} to {END}")
        
        for date in tqdm(monday_list):
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

def add_historical_data(csv_file, target_start_date):
    """Add historical data before the current earliest date in CSV."""
    first_date = get_first_date_from_csv(csv_file)
    if not first_date:
        print("No existing data found.")
        return
    
    # Calculate the last Monday before first_date
    days_back = (first_date.weekday() + 1) % 7
    if days_back == 0:  # first_date is Monday
        days_back = 7
    end_date = first_date - dt.timedelta(days=days_back)
    
    print(f"Current data starts from: {first_date}")
    print(f"Fetching historical data from {target_start_date} to {end_date}")
    
    # Scrape historical data
    historical_df = scrape_historical(start_date=target_start_date)
    
    # Filter to only include dates before first_date
    historical_df = historical_df[historical_df['snapshot_date'] < pd.to_datetime(first_date)]
    
    if not historical_df.empty:
        # Load existing data
        existing_df = pd.read_csv(csv_file)
        existing_df['snapshot_date'] = pd.to_datetime(existing_df['snapshot_date'])
        
        # Combine historical + existing data
        combined_df = pd.concat([historical_df, existing_df], ignore_index=True)
        combined_df = combined_df.sort_values('snapshot_date').reset_index(drop=True)
        
        # Save combined data
        combined_df.to_csv(csv_file, index=False)
        print(f"Added {len(historical_df)} historical rows. Total rows: {len(combined_df)}")
    else:
        print("No historical data to add.")

if __name__ == '__main__':
    import sys
    csv_file = "top100_weekly_data.csv"
    
    # Check for command line argument to add historical data
    if len(sys.argv) > 1 and sys.argv[1] == "--add-historical":
        # Add data from beginning of 2021
        target_start = dt.date(2021, 1, 4)  # First Monday of 2021
        add_historical_data(csv_file, target_start)
    else:
        # Normal operation - check for new data
        last_date = get_last_date_from_csv(csv_file)
        
        if last_date:
            # Calculate next Monday after last date
            next_monday = last_date + dt.timedelta(days=(7 - last_date.weekday()) % 7)
            if next_monday == last_date:  # if last_date is already Monday
                next_monday += dt.timedelta(days=7)
            
            print(f"Existing data found. Last date: {last_date}")
            print(f"Fetching new data from: {next_monday}")
            
            # Scrape only new data
            new_df = scrape_historical(start_date=next_monday)
            
            if not new_df.empty:
                # Load existing data and append new data
                existing_df = pd.read_csv(csv_file)
                existing_df['snapshot_date'] = pd.to_datetime(existing_df['snapshot_date'])
                
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.sort_values('snapshot_date').reset_index(drop=True)
                
                # Save combined data
                combined_df.to_csv(csv_file, index=False)
                print(f"Added {len(new_df)} new rows. Total rows: {len(combined_df)}")
            else:
                print("No new data to add.")
        else:
            # No existing file, scrape all data
            print("No existing data found. Scraping all historical data...")
            df = scrape_historical()
            df.to_csv(csv_file, index=False)
            print(f"Saved {len(df)} rows to {csv_file}")
