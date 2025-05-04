import pandas as pd
import requests
import datetime as dt
import time
import json
import sys
import os

class CryptoCompareFetcher:
    def __init__(self, api_key: str, limit: int = 100, tsym: str = "BTC",
                 request_timeout: int = 15, api_sleep_interval: int = 1,
                 top_n: int = 20, excluded_symbols: list[str] | None = None):
        self.api_key = api_key
        self.limit = limit
        self.tsym = tsym
        self.request_timeout = request_timeout
        self.api_sleep_interval = api_sleep_interval
        self.top_n = top_n
        self.excluded_symbols = excluded_symbols if excluded_symbols is not None else []
        self._base_url = "https://min-api.cryptocompare.com/data"

    def _process_raw_data(self, raw_data: list, btc_usd_price: float | None) -> pd.DataFrame:
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
            ValueError: If required columns ('CoinInfo_Name', f'RAW_{self.tsym}_MKTCAP',
                        f'RAW_{self.tsym}_PRICE') are missing in the normalized DataFrame.
        """
        if not raw_data:  # Empty data => return empty DataFrame
            return pd.DataFrame()

        df = pd.json_normalize(raw_data, sep='_')

        # Require price_btc for USD price calculation
        required_cols = {'CoinInfo_Name', f'RAW_{self.tsym}_MKTCAP', f'RAW_{self.tsym}_PRICE'}
        missing_cols = required_cols.difference(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Cannot process data.")

        df = (df
              .rename(columns={'CoinInfo_Name': 'sym',
                               f'RAW_{self.tsym}_MKTCAP': 'mcap_btc',
                               f'RAW_{self.tsym}_PRICE': 'price_btc'}) # Rename price_btc column
              .loc[lambda d: ~d.sym.isin(self.excluded_symbols)] # Exclude specified symbols
              .nlargest(self.top_n, 'mcap_btc')) # Select top N by market cap

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

        # Return DataFrame including the new USD price column and BTC/USD price
        df['btc_price_usd'] = btc_usd_price # Add the BTC/USD price as a new column
        return df[['sym', 'mcap_btc', 'price_btc', 'price_usd', 'btc_price_usd', 'weight']]

    def fetch_snapshot_data(self, snapshot_dt: dt.datetime) -> pd.DataFrame:
        """
        Fetches the top coins by market cap and historical BTC price for a specific
        snapshot timestamp from the CryptoCompare API and processes the data.

        Args:
            snapshot_dt: The datetime object representing the snapshot time.

        Returns:
            A pandas DataFrame containing the processed top coin data for the snapshot,
            including a 'rebalance_ts' column. Returns an empty DataFrame if the
            snapshot data is unavailable or processing fails.
        """
        ts = int(snapshot_dt.timestamp())

        # Fetch top coins data
        url_top_coins = (f"{self._base_url}/top/mktcapfull"
                         f"?limit={self.limit}&ts={ts}&tsym={self.tsym}&api_key={self.api_key}")

        print(f"Fetching top coins data for {snapshot_dt.isoformat()}...")
        try:
            resp_top_coins = requests.get(url_top_coins, timeout=self.request_timeout).json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Request failed for top coins data for {snapshot_dt.isoformat()}: {e}", file=sys.stderr)
            return pd.DataFrame()

        # Check if the 'Data' field exists and is not empty for top coins
        if not resp_top_coins.get('Data'):
            print(f"⚠️ Top coins snapshot data not available for: {snapshot_dt.isoformat()}")
            print(f"   Response: {json.dumps(resp_top_coins, indent=2)[:300]}...") # Print partial response
            return pd.DataFrame()

        # Fetch historical BTC price in USD
        url_btc_usd_price = (f"{self._base_url}/pricehistorical"
                             f"?fsym=BTC&tsyms=USD&ts={ts}&api_key={self.api_key}")

        print(f"Fetching BTC/USD price for {snapshot_dt.isoformat()}...")
        btc_usd_price = None
        try:
            resp_btc_usd_price = requests.get(url_btc_usd_price, timeout=self.request_timeout).json()
            # Extract BTC/USD price
            btc_usd_price = resp_btc_usd_price.get('BTC', {}).get('USD')
            if btc_usd_price is None:
                 print(f"⚠️ BTC/USD price not available for: {snapshot_dt.isoformat()}")

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Request failed for BTC/USD price for {snapshot_dt.isoformat()}: {e}", file=sys.stderr)
            # We can still try to process the top coins data even if BTC/USD price fetch fails


        try:
            # Pass BTC/USD price to _process_raw_data
            df = self._process_raw_data(resp_top_coins['Data'], btc_usd_price)
            df['rebalance_ts'] = snapshot_dt
            return df
        except ValueError as e: # Error during _process_raw_data processing
            print(f"⚠️ Error processing data for {snapshot_dt.isoformat()}: {e}", file=sys.stderr)
            print(f"   Response: {json.dumps(resp_top_coins, indent=2)[:600]}...", file=sys.stderr) # Print partial response
            return pd.DataFrame()
        except Exception as e: # Catch other potential errors during request/processing
            print(f"⚠️ An unexpected error occurred for {snapshot_dt.isoformat()}: {e}", file=sys.stderr)
            return pd.DataFrame()