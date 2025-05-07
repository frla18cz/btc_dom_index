"""
analyzer.py – Reálná implementace strategie:
- část long BTC (USD sizing)
- část short altcoinů (USD sizing jako short ALT/USD nebo BTC/ALT)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from portfolio_logger import PortfolioLogger
from config.config import OUTPUT_FILENAME

# --- PARAMETRY ---
CSV_PATH = Path(OUTPUT_FILENAME)
START_CAP = 100_000.0  # startovací kapitál v USD

BTC_W = 1.5      # např. 1.0 znamená long za 100 % kapitálu
ALT_W = 0.5      # např. 0.5 znamená short za 50 % kapitálu
TOP_N = 20
EXCLUDED = []    # mince k vyloučení (např. ['BTC', 'WBTC'])

SNAP_CSV = Path("portfolio_snapshots.csv")

# --- HLAVNÍ BACKTEST FUNKCE ---
def backtest_rank(df: pd.DataFrame,
                  btc_w: float = BTC_W,
                  alt_w: float = ALT_W,
                  top_n: int = TOP_N,
                  start_cap: float = START_CAP,
                  logger: PortfolioLogger | None = None) -> pd.DataFrame:

    weeks = sorted(df["rebalance_ts"].unique())
    equity = start_cap

    btc_qty = 0.0
    alt_qty = {}

    rows = []
    cum_btc_pnl = cum_alt_pnl = 0.0

    for i in range(len(weeks) - 1):
        t0, t1 = weeks[i], weeks[i + 1]
        w0, w1 = df[df.rebalance_ts == t0], df[df.rebalance_ts == t1]

        # --- první otevření pozic ---
        if i == 0:
            btc_price0 = w0["btc_price_usd"].iloc[0]
            btc_qty = (btc_w * equity) / btc_price0

            # short alty v USD
            alt_notional_usd = alt_w * equity
            alts = w0[~w0.sym.isin(EXCLUDED)].nsmallest(top_n, "rank")
            tot_mcap = alts["mcap_btc"].sum()
            alt_qty = {
                r.sym: -(alt_notional_usd * r.mcap_btc / tot_mcap) / r.price_usd
                for r in alts.itertuples()
            }
            continue

        # --- výpočet P/L za týden ---
        btc_price0 = w0["btc_price_usd"].iloc[0]
        btc_price1 = w1["btc_price_usd"].iloc[0]
        btc_pnl_usd = (btc_price1 - btc_price0) * btc_qty

        # alty v USD páru (ALT/USD)
        alt_pnl_usd = sum(
            (w1.loc[w1.sym == sym, "price_usd"].values[0] -
             w0.loc[w0.sym == sym, "price_usd"].values[0]) * qty
            for sym, qty in alt_qty.items()
            if not w1.loc[w1.sym == sym].empty and not w0.loc[w0.sym == sym].empty
        )

        cum_btc_pnl += btc_pnl_usd
        cum_alt_pnl += alt_pnl_usd
        equity += btc_pnl_usd + alt_pnl_usd

        print(f"{pd.Timestamp(t1).date()} | equity {equity:,.2f} USD "
              f"| BTC P/L {btc_pnl_usd:+,.2f} | ALT P/L {alt_pnl_usd:+,.2f}")

        # --- logger (volitelné) ---
        if logger:
            alt_prices = {r.sym: r.price_usd for r in w1.itertuples()}
            logger.record(
                date=pd.Timestamp(t1),
                equity_usd=equity,
                btc_qty=btc_qty,
                alt_qty=alt_qty,
                btc_price=btc_price1,
                alt_prices=alt_prices,
                btc_pnl=btc_pnl_usd,
                alt_pnl=alt_pnl_usd,
            )

        # --- re-balance ---
        btc_qty = (btc_w * equity) / btc_price1

        alt_notional_usd = alt_w * equity
        alts = w1[~w1.sym.isin(EXCLUDED)].nsmallest(top_n, "rank")
        tot_mcap = alts["mcap_btc"].sum()
        alt_qty = {
            r.sym: -(alt_notional_usd * r.mcap_btc / tot_mcap) / r.price_usd
            for r in alts.itertuples()
        }

        rows.append({
            "Date": pd.Timestamp(t1),
            "Equity_USD": equity,
            "BtcQty": btc_qty,
            "AltShortCount": len(alt_qty)
        })

    print("\n--- Souhrn ---")
    print(f"Kumulativní BTC P/L : {cum_btc_pnl:+,.2f} USD")
    print(f"Kumulativní ALT P/L : {cum_alt_pnl:+,.2f} USD")
    print(f"Konečná equity      : {equity:,.2f} USD")
    # přepočet konečné equity na ekvivalent BTC
    btc_equiv = equity / btc_price1
    print(f"Konečný ekvivalent  : {btc_equiv:,.6f} BTC\n")

    return pd.DataFrame(rows)

# --- MAIN ---
def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["rebalance_ts"] = pd.to_datetime(df["rebalance_ts"])

    logger = PortfolioLogger()

    perf = backtest_rank(df, logger=logger)

    logger.dump_csv(SNAP_CSV)

    # --- graf equity ---
    ymin = min(START_CAP, perf["Equity_USD"].min()) * 0.995
    ymax = perf["Equity_USD"].max() * 1.005
    fmt = ScalarFormatter(useOffset=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(perf["Date"], perf["Equity_USD"], marker="o")
    ax.set_title(f"{BTC_W:.1f}× BTC long | {ALT_W:.1f}× ALT short (top {TOP_N})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.4)
    ax.yaxis.set_major_formatter(fmt)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
