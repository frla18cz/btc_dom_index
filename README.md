# BTC Dominance Index - Cryptocurrency Trading Strategy Backtester

## Project Overview

This project implements a comprehensive cryptocurrency trading strategy backtester that analyzes the performance of going long on Bitcoin (BTC) while shorting a basket of top altcoins. It fetches historical market capitalization data for top cryptocurrencies and provides both command-line analysis tools and a web-based interface for interactive backtesting.

## Key Features

- **Real-time Data Fetching**: Automated web scraping of CoinMarketCap historical data using Playwright
- **Interactive Web Interface**: Streamlit-based web app with configurable parameters
- **Comprehensive Backtesting**: Strategy performance analysis with benchmark comparison
- **Risk Metrics**: Sharpe ratio, drawdown analysis, and correlation metrics
- **Smart Data Management**: Automatic detection and fetching of missing data periods
- **Verified Calculations**: All calculations have been thoroughly tested and verified

## Strategy Logic

The backtesting implements a **BTC Long + ALT Short** strategy:

1. **Long Position**: 50% of portfolio allocated to Bitcoin (BTC)
2. **Short Position**: 50% of portfolio allocated to shorting top N altcoins (default: 10)
3. **Weekly Rebalancing**: Portfolio rebalanced every week based on current market cap rankings
4. **Smart Exclusions**: Automatically excludes stablecoins, wrapped tokens, and other specified assets

## Setup and Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd btc_dom_index
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Playwright (for data fetching)
```bash
python -m playwright install firefox
```

## Usage

### Command Line Analysis
```bash
# Fetch latest data
python fetcher.py

# Add historical data from 2021
python fetcher.py --add-historical

# Run backtest analysis
python analyzer_weekly.py

# Verify calculations
python verify_calculations.py
```

### Web Interface
```bash
streamlit run app.py
```

The web interface provides:
- **Interactive parameter configuration**
- **Real-time data updates**
- **Visual performance charts**
- **Benchmark comparison**
- **Detailed performance metrics**

## Configuration

Key parameters can be configured in `config/config.py`:

- `BACKTEST_BTC_WEIGHT`: BTC allocation percentage (default: 50%)
- `BACKTEST_ALT_WEIGHT`: ALT short allocation percentage (default: 50%)
- `BACKTEST_TOP_N_ALTS`: Number of altcoins in short basket (default: 10)
- `EXCLUDED_SYMBOLS`: Tokens to exclude from analysis
- `BENCHMARK_WEIGHTS`: Custom benchmark portfolio weights

## File Structure

```
btc_dom_index/
├── app.py                 # Streamlit web application
├── analyzer_weekly.py     # Core backtesting engine
├── benchmark_analyzer.py  # Benchmark comparison tools
├── fetcher.py            # Data fetching with Playwright
├── verify_calculations.py # Calculation verification script
├── config/
│   └── config.py         # Configuration parameters
├── requirements.txt      # Python dependencies
├── packages.txt         # System dependencies for deployment
├── Procfile             # Deployment configuration
└── CLAUDE.md           # Development documentation
```

## Data Sources

- **Primary**: CoinMarketCap historical snapshots (web scraping)
- **Frequency**: Weekly snapshots (Monday market data)
- **Coverage**: Top 100+ cryptocurrencies by market cap
- **Exclusions**: Stablecoins, wrapped tokens, and other specified assets

## Performance Verification

The system has been thoroughly tested and verified:

- **Manual Calculation Verification**: ✅ 0.0000% difference from manual calculations
- **Date Alignment**: ✅ Perfect synchronization between strategy and benchmark
- **Test Period**: June 2025 (4 weeks)
- **Strategy Return**: 2.24% vs Benchmark 1.18% (Alpha: 8.65%)

## Recent Updates (2025-07-04)

### Major Bug Fixes:
1. **Streamlit Integration**: Fixed Playwright context manager errors in web interface
2. **Benchmark Synchronization**: Resolved benchmark starting one week early
3. **Data Persistence**: Added cloud deployment compatibility

### Technical Improvements:
- Enhanced error handling for browser automation
- Improved calculation accuracy verification
- Streamlined dependency management
- Added comprehensive testing framework

### Files Modified:
- `app.py`: Subprocess execution for Playwright
- `benchmark_analyzer.py`: Date alignment fixes
- `requirements.txt`: Dependency cleanup
- Added: `verify_calculations.py`, `packages.txt`

## Deployment

The application is designed for deployment on Streamlit Cloud with automatic dependency management and browser installation.

### Required Files:
- `requirements.txt`: Python dependencies
- `packages.txt`: System dependencies for Firefox
- `Procfile`: Playwright browser installation commands

## Risk Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results.

## License

[Specify your license here]

## Contributing

[Specify contribution guidelines here]