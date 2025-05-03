import pandas as pd
import requests
import datetime as dt
import time
import json
import sys
import os
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()  # Load variables from .env file
API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
if not API_KEY:
    raise ValueError("CRYPTOCOMPARE_API_KEY environment variable not set.")

LIMIT = 100  # Max allowed by CryptoCompare API for this endpoint
TSYM = "BTC"  # Target symbol for market cap calculation
START = dt.datetime(2025, 2, 17, 8)  # Start date for fetching data
END = dt.datetime(2025, 4, 24, 8)  # End date for fetching data
EXCLUDED_SYMBOLS = ['BTC', 'USDT', 'USDC', 'DAI', 'BUSD'] # Symbols to exclude from the top 20
OUTPUT_FILENAME = "btcdom_top20_weights.csv"
REQUEST_TIMEOUT = 15 # seconds
API_SLEEP_INTERVAL = 1 # seconds between API calls
TOP_N = 20 # Number of top coins to select


# --- Function 1: Process raw JSON data into a DataFrame ---
def to_dataframe(raw_data: list) -> pd.DataFrame:
    """
    Processes raw data from the CryptoCompare API response into a pandas DataFrame.

    Args:
        raw_data: A list of dictionaries, typically from the 'Data' field
                  of the CryptoCompare API response.

    Returns:
        A pandas DataFrame containing the top N coins (excluding specified symbols)
        ranked by market cap in TSYM, with calculated weights. Returns an empty
        DataFrame if raw_data is empty.

    Raises:
        ValueError: If required columns ('CoinInfo_Name', f'RAW_{TSYM}_MKTCAP')
                    are missing in the normalized DataFrame.
    """
    if not raw_data:  # Empty data => return empty DataFrame
        return pd.DataFrame()

    df = pd.json_normalize(raw_data, sep='_')

    req = {'CoinInfo_Name', f'RAW_{TSYM}_MKTCAP'}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    df = (df
          .rename(columns={'CoinInfo_Name': 'sym',
                           f'RAW_{TSYM}_MKTCAP': 'mcap_btc'})
          .loc[lambda d: ~d.sym.isin(EXCLUDED_SYMBOLS)] # Exclude specified symbols
          .nlargest(TOP_N, 'mcap_btc')) # Select top N by market cap
    
    # Calculate weights relative to the selected top N
    total_mcap = df.mcap_btc.sum()
    if total_mcap > 0:
        df['weight'] = df.mcap_btc / total_mcap
    else:
        df['weight'] = 0.0 # Avoid division by zero if total market cap is zero

    return df[['sym', 'mcap_btc', 'weight']]

# --- Function 2: Fetch a single weekly snapshot ---
def fetch_top20(snapshot_dt: dt.datetime) -> pd.DataFrame:
    """
    Fetches the top coins by market cap for a specific snapshot timestamp
    from the CryptoCompare API and processes the data.

    Args:
        snapshot_dt: The datetime object representing the snapshot time.

    Returns:
        A pandas DataFrame containing the processed top coin data for the snapshot,
        including a 'rebalance_ts' column. Returns an empty DataFrame if the
        snapshot data is unavailable or processing fails.
    """
    ts = int(snapshot_dt.timestamp())
    url = (f"https://min-api.cryptocompare.com/data/top/mktcapfull"
           f"?limit={LIMIT}&ts={ts}&tsym={TSYM}&api_key={API_KEY}")
    
    print(f"Fetching data for {snapshot_dt.isoformat()}...")
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT).json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Request failed for {snapshot_dt.isoformat()}: {e}", file=sys.stderr)
        return pd.DataFrame()

    # Check if the 'Data' field exists and is not empty
    if not resp.get('Data'):
        print(f"⚠️ Snapshot data not available for: {snapshot_dt.isoformat()}")
        print(f"   Response: {json.dumps(resp, indent=2)[:300]}...") # Print partial response
        return pd.DataFrame()

    try:
        df = to_dataframe(resp['Data'])
        df['rebalance_ts'] = snapshot_dt
        return df
    except ValueError as e: # Error during to_dataframe processing
        print(f"⚠️ Error processing data for {snapshot_dt.isoformat()}: {e}", file=sys.stderr)
        print(f"   Response: {json.dumps(resp, indent=2)[:600]}...", file=sys.stderr) # Print partial response
        return pd.DataFrame()
    except Exception as e: # Catch other potential errors during request/processing
        print(f"⚠️ An unexpected error occurred for {snapshot_dt.isoformat()}: {e}", file=sys.stderr)
        return pd.DataFrame()

# --- Main Loop: Iterate through weekly snapshots ---
print(f"Starting data fetch from {START.isoformat()} to {END.isoformat()}")
frames, current_time = [], START
while current_time <= END:
    snapshot_df = fetch_top20(current_time)
    if not snapshot_df.empty:
        frames.append(snapshot_df)
    
    current_time += dt.timedelta(days=7)
    # Respect API rate limits (adjust sleep time if necessary)
    time.sleep(API_SLEEP_INTERVAL)

weights = pd.concat(frames, ignore_index=True)
weights.to_csv(OUTPUT_FILENAME, index=False)
print(f"✔️ Done – saved rows: {len(weights)} to {OUTPUT_FILENAME}")
