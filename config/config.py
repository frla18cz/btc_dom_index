import datetime as dt

# --- Data Fetching Configuration ---
LIMIT = 100  # Max allowed by CryptoCompare API for this endpoint
TSYM = "BTC"  # Target symbol for market cap calculation
REQUEST_TIMEOUT = 15 # seconds
API_SLEEP_INTERVAL = 1 # seconds between API calls
TOP_N = 50 # Number of top coins to select

# Symbols to exclude from analysis (stablecoins, wrapped tokens, etc.)
EXCLUDED_SYMBOLS = [
    # Bitcoin and wrapped versions
    'BTC', 'WBTC', 'BTCB',
    
    # Stablecoins
    'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'USDD', 'USDE', 'USDS', 'FDUSD', 'USDP', 'GUSD', 'USDK',
    'LUSD', 'FRAX', 'sUSD', 'USDN', 'EURC', 'EURT', 'EUROC', 'PYUSD',
    
    # Wrapped ETH variants
    'WETH', 'WBETH', 'WEETH', 'STETH', 'FRXETH', 'CBETH',
    
    # Other wrapped or tokenized assets
    'PAXG', 'WTRX', 'WBNB', 'WMATIC', 'WAVAX' 
]

# --- Output Configuration ---
OUTPUT_FILENAME = "btcdom_top50_weights.csv"

# --- Time Range Configuration ---
# Default dates for data fetching
START = dt.datetime(2025, 1, 1, 8)  # Start date for fetching data
END = dt.datetime(2025, 5, 1, 8)  # End date for fetching data

# --- Backtest Configuration ---
# Default dates for backtesting (can be overridden in analyzer)
BACKTEST_START_DATE = dt.datetime(2025, 1, 10)  # Monday after first snapshot
BACKTEST_END_DATE = dt.datetime(2025, 5, 1)     # End date for backtesting

# Strategy parameters
BACKTEST_INITIAL_CAPITAL = 100_000.0  # Starting capital in USD
BACKTEST_BTC_WEIGHT = 1.3            # Weight of BTC in the portfolio (0-1)
BACKTEST_ALT_WEIGHT = 0.3            # Weight of alts in the portfolio (0-1)
BACKTEST_TOP_N_ALTS = 10            # Number of top alts to include in the short basket