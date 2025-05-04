import pandas as pd
import datetime as dt
import time
import os
from dotenv import load_dotenv

# Import configuration and fetcher
from config.config import (
    LIMIT, TSYM, REQUEST_TIMEOUT, API_SLEEP_INTERVAL, TOP_N,
    EXCLUDED_SYMBOLS, OUTPUT_FILENAME, START, END, DB_PATH
)
from fetcher import CryptoCompareFetcher
from storage import Storage

# --- Main Execution Block ---
if __name__ == "__main__":
    load_dotenv()  # Load variables from .env file
    API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
    if not API_KEY:
        raise ValueError("CRYPTOCOMPARE_API_KEY environment variable not set.")

    print(f"Starting data fetch from {START.isoformat()} to {END.isoformat()}")

    fetcher = CryptoCompareFetcher(
        api_key=API_KEY,
        limit=LIMIT,
        tsym=TSYM,
        request_timeout=REQUEST_TIMEOUT,
        api_sleep_interval=API_SLEEP_INTERVAL,
        top_n=TOP_N,
        excluded_symbols=EXCLUDED_SYMBOLS
    )

    # Initialize persistent storage for snapshots and token metadata
    storage = Storage(DB_PATH)
    frames = []
    current_time = START
    while current_time <= END:
        snapshot_df = fetcher.fetch_snapshot_data(current_time)
        if not snapshot_df.empty:
            # Persist snapshot to database
            storage.store_snapshot(snapshot_df)
            frames.append(snapshot_df)

        current_time += dt.timedelta(days=7)
        # Respect API rate limits (adjust sleep time if necessary)
        time.sleep(fetcher.api_sleep_interval)

    if frames:
        weights = pd.concat(frames, ignore_index=True)
        weights.to_csv(OUTPUT_FILENAME, index=False)
        print(f"✔️ Done – saved rows: {len(weights)} to {OUTPUT_FILENAME}")
    else:
        print("❌ No data fetched.")
    # Close storage connection
    storage.close()
