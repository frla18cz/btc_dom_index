# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BTC Dominance Index is a cryptocurrency analysis tool that fetches historical market capitalization data for top cryptocurrencies (excluding BTC and major stablecoins) and implements a backtesting strategy of going long on Bitcoin (BTC) while shorting a basket of altcoins.

## Setup and Environment

1. **Virtual Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Playwright Setup** (for fetcher.py)
   ```bash
   # Install Playwright browsers
   python -m playwright install firefox
   ```

## Command Reference

### Data Fetching

**Run the Playwright-based fetcher to scrape CoinMarketCap data**
```bash
python fetcher.py
```
This will automatically detect existing data and fetch only new snapshots from the last available date to today.

**To add historical data from earlier periods (e.g., beginning of 2021)**
```bash
python fetcher.py --add-historical
```
This will add data from January 2021 to the earliest date in your existing dataset.

The fetcher intelligently:
- Detects existing data range in `top100_weekly_data.csv`
- Fetches only missing data (either new or historical)
- Merges new data with existing data automatically
- Prevents duplicate snapshots

### Analysis and Backtesting

**Run the weekly analyzer to backtest the BTC-long/ALT-short strategy**
```bash
python analyzer_weekly.py
```
This will load data from `top100_weekly_data.csv`, run a backtest of the strategy, and display a performance chart.

### Streamlit Web App

**Run the Streamlit web app for interactive backtesting**
```bash
streamlit run app.py
```
This will launch a web interface that allows you to configure and run the backtesting strategy with custom parameters.

The web app includes:
- **Smart date selection**: Shows available data range and prevents invalid date selection
- **Visual feedback**: Clear indicators of data availability and period selection
- **Real-time validation**: Checks if selected dates have data available
- **Automatic data updates**: Built-in data management with one-click updates for missing data
- **Comprehensive configuration**: Leverage, altcoin basket size, excluded tokens

#### Automatic Data Management
The Streamlit app automatically:
- Detects when new weekly data is available
- Shows a notification in the sidebar with the number of missing snapshots
- Provides a "🔄 Update Data" button to download missing data directly from the UI
- Updates the dataset and refreshes the interface automatically

## Code Architecture

The project follows a modular architecture:

1. **Data Fetching Layer**
   - `fetcher.py`: Implementation using Playwright for web scraping CoinMarketCap historical snapshots

2. **Configuration Layer**
   - `config/config.py`: Central configuration for parameters such as excluded tokens, date ranges, and API settings

3. **Analysis Layer**
   - `analyzer_weekly.py`: Implements the backtesting strategy with these key components:
     - `load_and_prepare()`: Data cleaning and preparation
     - `backtest_rank_altbtc_short()`: Core backtesting algorithm
     - Visualization code for plotting equity curves

4. **Web Interface Layer**
   - `app.py`: Streamlit web application that provides an interactive UI for the backtesting system
     - Configurable parameters through the sidebar
     - Visual presentation of backtest results
     - Multiple views including summary, charts, and detailed data

## Strategy Logic

The backtesting algorithm in `analyzer_weekly.py` implements:

1. Weekly rebalancing of a portfolio that:
   - Goes long on Bitcoin (BTC) with `BTC_W` portion of equity (default 50%)
   - Goes short on a basket of top `TOP_N` altcoins (default 10) with `ALT_W` portion of equity (default 50%)
   - Excludes specific tokens defined in `EXCLUDED` (e.g., stablecoins, wrapped tokens)

2. Weights for the short altcoin basket are determined by market capitalization
   - Each altcoin's weight is proportional to its market cap within the selected basket

3. Performance is tracked in both USD and BTC terms