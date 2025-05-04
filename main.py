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
def to_dataframe(raw_data: list, btc_usd_price: float | None) -> pd.DataFrame:
    """
    Processes raw data from the CryptoCompare API response into a pandas DataFrame,
    calculating USD prices if BTC/USD price is available.

    Args:
        raw_data: A list of dictionaries, typically from the 'Data' field
                  of the CryptoCompare API response.
        btc_usd_price: The historical price of BTC in USD for the snapshot time,
                       or None if not available.

    Returns:
        A pandas DataFrame containing the top N coins (excluding specified symbols)
        ranked by market cap in TSYM, with calculated weights and USD prices.
        Returns an empty DataFrame if raw_data is empty.

    Raises:
        ValueError: If required columns ('CoinInfo_Name', f'RAW_{TSYM}_MKTCAP',
                    f'RAW_{TSYM}_PRICE') are missing in the normalized DataFrame.
    """
    if not raw_data:  # Empty data => return empty DataFrame
        return pd.DataFrame()

    df = pd.json_normalize(raw_data, sep='_')

    # Require price_btc for USD price calculation
    req = {'CoinInfo_Name', f'RAW_{TSYM}_MKTCAP', f'RAW_{TSYM}_PRICE'}
    miss = req.difference(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {miss}. Cannot calculate USD prices.")

    df = (df
          .rename(columns={'CoinInfo_Name': 'sym',
                           f'RAW_{TSYM}_MKTCAP': 'mcap_btc',
                           f'RAW_{TSYM}_PRICE': 'price_btc'}) # Rename price_btc column
          .loc[lambda d: ~d.sym.isin(EXCLUDED_SYMBOLS)] # Exclude specified symbols
          .nlargest(TOP_N, 'mcap_btc')) # Select top N by market cap
    
    # Calculate weights relative to the selected top N
    total_mcap = df.mcap_btc.sum()
    if total_mcap > 0:
        df['weight'] = df.mcap_btc / total_mcap
    else:
        df['weight'] = 0.0 # Avoid division by zero if total market cap is zero

    # Calculate USD price if BTC/USD price is available
    if btc_usd_price is not None:
        df['price_usd'] = df['price_btc'] * btc_usd_price
    else:
        df['price_usd'] = None # Set to None if BTC/USD price is not available

    # Return DataFrame including the new USD price column
    return df[['sym', 'mcap_btc', 'price_btc', 'price_usd', 'weight']]

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
    
    # Fetch top coins data
    url_top_coins = (f"https://min-api.cryptocompare.com/data/top/mktcapfull"
                     f"?limit={LIMIT}&ts={ts}&tsym={TSYM}&api_key={API_KEY}")
    
    print(f"Fetching top coins data for {snapshot_dt.isoformat()}...")
    try:
        resp_top_coins = requests.get(url_top_coins, timeout=REQUEST_TIMEOUT).json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Request failed for top coins data for {snapshot_dt.isoformat()}: {e}", file=sys.stderr)
        return pd.DataFrame()

    # Check if the 'Data' field exists and is not empty for top coins
    if not resp_top_coins.get('Data'):
        print(f"⚠️ Top coins snapshot data not available for: {snapshot_dt.isoformat()}")
        print(f"   Response: {json.dumps(resp_top_coins, indent=2)[:300]}...") # Print partial response
        return pd.DataFrame()

    # Fetch historical BTC price in USD
    url_btc_usd_price = (f"https://min-api.cryptocompare.com/data/pricehistorical"
                         f"?fsym=BTC&tsyms=USD&ts={ts}&api_key={API_KEY}")

    print(f"Fetching BTC/USD price for {snapshot_dt.isoformat()}...")
    try:
        resp_btc_usd_price = requests.get(url_btc_usd_price, timeout=REQUEST_TIMEOUT).json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Request failed for BTC/USD price for {snapshot_dt.isoformat()}: {e}", file=sys.stderr)
        # We can still try to process the top coins data even if BTC/USD price fetch fails
        btc_usd_price = None
    
    # Extract BTC/USD price
    btc_usd_price = resp_btc_usd_price.get('BTC', {}).get('USD') if resp_btc_usd_price else None
    
    if btc_usd_price is None:
         print(f"⚠️ BTC/USD price not available for: {snapshot_dt.isoformat()}")
         # We can still try to process the top coins data even if BTC/USD price is missing
         
    try:
        # Pass BTC/USD price to to_dataframe
        df = to_dataframe(resp_top_coins['Data'], btc_usd_price)
        df['rebalance_ts'] = snapshot_dt
        return df
    except ValueError as e: # Error during to_dataframe processing
        print(f"⚠️ Error processing data for {snapshot_dt.isoformat()}: {e}", file=sys.stderr)
        print(f"   Response: {json.dumps(resp_top_coins, indent=2)[:600]}...", file=sys.stderr) # Print partial response
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
