import datetime as dt

# --- Data Fetching Configuration ---
LIMIT = 100  # Max allowed by CryptoCompare API for this endpoint
TSYM = "BTC"  # Target symbol for market cap calculation
REQUEST_TIMEOUT = 15 # seconds
API_SLEEP_INTERVAL = 1 # seconds between API calls
TOP_N = 50 # Number of top coins to select
EXCLUDED_SYMBOLS = ['BTC', 'USDT', 'USDC', 'DAI', 'BUSD','USDE', 'USDS', 'WBETH', 'WEETH'] # Symbols to exclude from the top 20

# --- Output Configuration ---
OUTPUT_FILENAME = "btcdom_top50_weights.csv"
DB_PATH = "btcdom_data.db"

# --- Time Range Configuration ---
START = dt.datetime(2025, 2, 17, 8)  # Start date for fetching data
END = dt.datetime(2025, 4, 24, 8)  # End date for fetching data