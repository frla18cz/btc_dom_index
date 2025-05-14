# analyzer.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
# Importujeme upravený logger
from portfolio_logger import PortfolioLogger
# Předpokládáme, že config.py existuje a obsahuje OUTPUT_FILENAME
from config.config import OUTPUT_FILENAME

# --- PARAMETRY ---
CSV_PATH = Path(OUTPUT_FILENAME)
START_CAP = 100_000.0
BTC_W = 0.5
ALT_W = 0.5
TOP_N = 10  # Změněno na 10 pro shodu s tvým výstupem, uprav podle potřeby
EXCLUDED = ['BTC', 'WBTC']  # ETH a stETH mohou být nyní zahrnuty

SNAP_CSV = Path("portfolio_snapshots_altbtc_short_detailed.csv")


# --- HLAVNÍ BACKTEST FUNKCE ---
def backtest_rank_altbtc_short(df: pd.DataFrame,
                               btc_w: float = BTC_W,
                               alt_w: float = ALT_W,
                               top_n: int = TOP_N,
                               start_cap: float = START_CAP,
                               logger: PortfolioLogger | None = None) -> tuple[pd.DataFrame, dict]:
    weeks = sorted(df["rebalance_ts"].unique())
    equity = start_cap

    btc_qty = 0.0
    alt_qty = {}

    rows = []
    cum_btc_pnl = cum_alt_pnl = 0.0
    total_alt_pnl_btc = 0.0

    print("--- Backtest Start ---")
    print(f"Strategy: {btc_w * 100}% Long BTC/USD, {alt_w * 100}% Short ALT/BTC (Top {top_n})")
    print(f"Excluded from Alts: {EXCLUDED}")
    print(f"Initial Equity: {start_cap:,.2f} USD")
    print("-" * 50)

    for i in range(len(weeks) - 1):
        t0, t1 = weeks[i], weeks[i + 1]
        w0 = df[df.rebalance_ts == t0].set_index('sym')
        w1 = df[df.rebalance_ts == t1].set_index('sym')

        try:
            btc_price0 = w0["btc_price_usd"].mean()
            btc_price1 = w1["btc_price_usd"].mean()
            if pd.isna(btc_price0) or pd.isna(btc_price1) or btc_price0 == 0:
                raise ValueError(f"BTC price is NaN or zero at t0 ({t0}) or t1 ({t1})")
        except Exception as e:
            print(f"Error getting BTC price: {e}")
            print(f"Skipping week {pd.Timestamp(t0).date()} -> {pd.Timestamp(t1).date()}.")
            if i == 0:
                print("Cannot initialize positions without BTC price.")
                return pd.DataFrame(), {}
            else:
                print("Warning: Holding previous positions due to missing BTC price.")
                if rows:
                    last_row = rows[-1]
                    rows.append({**last_row, 'Date': pd.Timestamp(t1), 'BTC_Price_USD': None})
                continue

        if i == 0:
            print(f"{pd.Timestamp(t0).date()} | Initializing positions...")
            btc_qty = (btc_w * equity) / btc_price0
            print(f"  BTC Long: Qty={btc_qty:.6f} BTC @ ${btc_price0:,.2f}/BTC | Value=${btc_qty * btc_price0:,.2f}")

            alt_notional_usd_target = alt_w * equity
            alts_df_t0 = w0[~w0.index.isin(EXCLUDED)].nsmallest(top_n, "rank")
            actual_alt_usd_value_total = 0

            if not alts_df_t0.empty:
                tot_mcap = alts_df_t0["mcap_btc"].sum()
                if tot_mcap > 0 and not pd.isna(tot_mcap):
                    print(f"  Target ALT Short Leg Value: ${alt_notional_usd_target:,.2f}")
                    print(f"  Selected {len(alts_df_t0)} Alts for Shorting:")
                    temp_alt_qty = {}
                    for sym, r in alts_df_t0.iterrows():
                        if not pd.isna(r.price_btc) and r.price_btc != 0 and not pd.isna(r.mcap_btc):
                            alt_usd_target_sym = alt_notional_usd_target * (r.mcap_btc / tot_mcap)
                            qty = -(alt_usd_target_sym / (r.price_btc * btc_price0))
                            temp_alt_qty[sym] = qty
                            actual_usd_value_sym = abs(qty * r.price_btc * btc_price0)
                            actual_alt_usd_value_total += actual_usd_value_sym
                            print(
                                f"    {sym}: Weight={r.mcap_btc / tot_mcap:.2%}, Target=${alt_usd_target_sym:,.2f}, Qty={qty:,.4f}, PriceBTC={r.price_btc:.8f}, ActualValue=${actual_usd_value_sym:,.2f}")
                        else:
                            print(f"  Warning: Skipping {sym} in init due to zero/NaN price_btc or mcap_btc.")
                    alt_qty = temp_alt_qty
                    print(f"  Total Actual ALT Short Value Initialized: ${actual_alt_usd_value_total:,.2f}")
                else:
                    print(f"  Warning: Total mcap_btc is zero or NaN for selected alts at {t0}. Cannot short.")
                    alt_qty = {}
            else:
                print("  Warning: No altcoins available for shorting basket at init.")
                alt_qty = {}

            rows.append({
                "Date": pd.Timestamp(t0), "Equity_USD": equity, "BTC_Price_USD": btc_price0,
                "BtcQty": btc_qty, "BtcHold_USD": btc_qty * btc_price0,
                "AltShortTarget_USD": alt_notional_usd_target,
                "AltShortActual_USD": actual_alt_usd_value_total,
                "AltShortCount": len(alt_qty), "Cum_BTC_PNL_USD": 0, "Cum_ALT_PNL_USD": 0,
                "Cum_ALT_PNL_BTC": 0,
            })
            print("-" * 50)
            continue

        # --- Výpočet P/L ---
        print(f"--- Calculating P/L for week {pd.Timestamp(t0).date()} -> {pd.Timestamp(t1).date()} ---")
        btc_pnl_usd = (btc_price1 - btc_price0) * btc_qty
        print(
            f"  BTC Position: Qty={btc_qty:.6f}, Price_Start=${btc_price0:,.2f}, Price_End=${btc_price1:,.2f}, PNL_USD={btc_pnl_usd:+,.2f}")

        weekly_alt_pnl_btc = 0.0
        print(f"  Calculating ALT/BTC PNL details:")

        for sym, qty_held in alt_qty.items():
            pnl_btc_sym = 0.0
            pnl_usd_sym = 0.0

            if sym in w0.index and sym in w1.index:
                price_btc0_sym = w0.loc[sym, "price_btc"]
                price_btc1_sym = w1.loc[sym, "price_btc"]

                if not pd.isna(price_btc0_sym) and not pd.isna(price_btc1_sym):
                    pnl_btc_sym = (price_btc1_sym - price_btc0_sym) * qty_held

                    if not pd.isna(btc_price1) and btc_price1 != 0:
                        pnl_usd_sym = pnl_btc_sym * btc_price1
                    else:
                        pnl_usd_sym = 0

                    weekly_alt_pnl_btc += pnl_btc_sym

                    print(f"    {sym}: HeldQty={qty_held:,.4f}, "
                          f"PriceBTC_Start={price_btc0_sym:.8f}, PriceBTC_End={price_btc1_sym:.8f}, "
                          f"PNL_BTC_Sym={pnl_btc_sym:+.8f}, PNL_USD_Sym={pnl_usd_sym:+,.2f}")
                else:
                    print(
                        f"    {sym}: HeldQty={qty_held:,.4f}, PriceBTC_Start/End missing. PNL for sym not calculated.")
            else:
                print(
                    f"    {sym}: HeldQty={qty_held:,.4f}, Data missing in w0 or w1. PNL for sym assumed 0 for this period.")

        if pd.isna(btc_price1) or btc_price1 == 0:
            print(f"Error: BTC price is NaN or zero at {t1}. Cannot calculate total USD PNL for alts or rebalance.")
            if rows:
                last_row = rows[-1]
                rows.append({**last_row, 'Date': pd.Timestamp(t1), 'BTC_Price_USD': None})
            continue

        weekly_alt_pnl_usd = weekly_alt_pnl_btc * btc_price1

        cum_btc_pnl += btc_pnl_usd
        cum_alt_pnl += weekly_alt_pnl_usd
        total_alt_pnl_btc += weekly_alt_pnl_btc
        equity += btc_pnl_usd + weekly_alt_pnl_usd

        if pd.isna(equity) or equity <= 0:
            print(f"Error: Equity became NaN or non-positive ({equity}) at {t1}. Stopping backtest.")
            break

        print(f"  Summary for {pd.Timestamp(t1).date()}: Equity {equity:,.2f} USD "
              f"| Week BTC P/L {btc_pnl_usd:+,.2f} USD | Week ALT P/L {weekly_alt_pnl_usd:+,.2f} USD "
              f"({weekly_alt_pnl_btc:+.6f} BTC)")

        if logger:
            alt_prices_usd_t1 = w1['price_usd'].to_dict()
            alt_prices_btc_t1 = w1['price_btc'].to_dict()
            current_alt_prices_usd = {s: alt_prices_usd_t1.get(s) for s in alt_qty.keys()}
            current_alt_prices_btc = {s: alt_prices_btc_t1.get(s) for s in alt_qty.keys()}
            try:
                logger.record_altbtc(
                    date=pd.Timestamp(t1), equity_usd=equity, btc_qty=btc_qty, alt_qty=alt_qty,
                    btc_price_usd=btc_price1, alt_prices_usd=current_alt_prices_usd,
                    alt_prices_btc=current_alt_prices_btc, btc_pnl_usd=btc_pnl_usd,
                    alt_pnl_usd=weekly_alt_pnl_usd, alt_pnl_btc=weekly_alt_pnl_btc
                )
            except Exception as e:
                print(f"  Error during logging: {e}")

        print(f"  Rebalancing for date {pd.Timestamp(t1).date()}...")
        btc_qty_old = btc_qty  # Pro případný výpis změny
        btc_qty = (btc_w * equity) / btc_price1
        print(
            f"  BTC Long: OldQty={btc_qty_old:.6f}, NewQty={btc_qty:.6f} @ ${btc_price1:,.2f}/BTC | New Value=${btc_qty * btc_price1:,.2f}")

        alt_notional_usd_target = alt_w * equity
        old_alt_symbols = set(alt_qty.keys())
        alts_df_t1 = w1[~w1.index.isin(EXCLUDED)].nsmallest(top_n, "rank")

        actual_alt_usd_value_total = 0
        new_alt_qty = {}

        if not alts_df_t1.empty:
            tot_mcap = alts_df_t1["mcap_btc"].sum()
            if tot_mcap > 0 and not pd.isna(tot_mcap):
                current_alt_symbols_target = set(alts_df_t1.index)
                removed_symbols = old_alt_symbols - current_alt_symbols_target
                added_symbols = current_alt_symbols_target - old_alt_symbols

                if removed_symbols:
                    print(f"  Alts Removed from Short Basket: {', '.join(sorted(list(removed_symbols)))}")
                if added_symbols:
                    print(f"  Alts Added to Short Basket: {', '.join(sorted(list(added_symbols)))}")
                if not removed_symbols and not added_symbols and old_alt_symbols == current_alt_symbols_target and len(
                        old_alt_symbols) > 0:
                    print(f"  Alt Short Basket Composition Unchanged (Top {len(current_alt_symbols_target)}).")
                elif not old_alt_symbols and not current_alt_symbols_target and len(
                        alts_df_t1) == 0:  # Pokud je koš prázdný
                    print(f"  Alt Short Basket remains empty.")
                elif not old_alt_symbols and current_alt_symbols_target:  # Pokud byl prázdný a teď se plní
                    print(
                        f"  Initializing Alt Short Basket with: {', '.join(sorted(list(current_alt_symbols_target)))}")

                print(f"  New Target ALT Short Leg Value: ${alt_notional_usd_target:,.2f}")
                print(f"  Rebalancing details for {len(alts_df_t1)} Alts:")

                for sym, r in alts_df_t1.iterrows():
                    if not pd.isna(r.price_btc) and r.price_btc != 0 and not pd.isna(r.mcap_btc):
                        alt_usd_target_sym = alt_notional_usd_target * (r.mcap_btc / tot_mcap)
                        qty = -(alt_usd_target_sym / (r.price_btc * btc_price1))
                        new_alt_qty[sym] = qty
                        actual_usd_value_sym = abs(qty * r.price_btc * btc_price1)
                        actual_alt_usd_value_total += actual_usd_value_sym
                        print(
                            f"    {sym}: TargetVal=${alt_usd_target_sym:,.2f}, NewQty={qty:,.4f}, PriceBTC={r.price_btc:.8f}, ActualVal=${actual_usd_value_sym:,.2f}")
                    else:
                        print(f"  Warning: Skipping {sym} in rebalance due to zero/NaN price_btc or mcap_btc.")

                alt_qty = new_alt_qty
                print(f"  Total Actual ALT Short Value after Rebalance: ${actual_alt_usd_value_total:,.2f}")
            else:
                print(
                    f"  Warning: Total mcap_btc is zero or NaN for selected alts at {t1}. Closing ALT short positions.")
                alt_qty = {}
        else:
            print("  Warning: No altcoins available for shorting basket during rebalance. Closing ALT short positions.")
            alt_qty = {}

        rows.append({
            "Date": pd.Timestamp(t1), "Equity_USD": equity, "BTC_Price_USD": btc_price1,
            "BtcQty": btc_qty, "BtcHold_USD": btc_qty * btc_price1,
            "AltShortTarget_USD": alt_notional_usd_target,
            "AltShortActual_USD": actual_alt_usd_value_total,
            "AltShortCount": len(alt_qty), "Cum_BTC_PNL_USD": cum_btc_pnl,
            "Cum_ALT_PNL_USD": cum_alt_pnl, "Cum_ALT_PNL_BTC": total_alt_pnl_btc,
        })
        print("-" * 50)

    print("\n--- Backtest End ---")
    last_btc_price = rows[-1]['BTC_Price_USD'] if rows and 'BTC_Price_USD' in rows[-1] and not pd.isna(
        rows[-1]['BTC_Price_USD']) else None
    btc_equiv = equity / last_btc_price if last_btc_price and last_btc_price != 0 else 0

    print(f"Cumulative BTC P/L : {cum_btc_pnl:+,.2f} USD")
    print(f"Cumulative ALT P/L : {cum_alt_pnl:+,.2f} USD ({total_alt_pnl_btc:+.6f} BTC)")
    print(f"Final equity       : {equity:,.2f} USD")
    print(f"Final equivalent   : {btc_equiv:,.6f} BTC\n")

    perf_df = pd.DataFrame(rows)
    summary = {
        "cum_btc_pnl": cum_btc_pnl, "cum_alt_pnl": cum_alt_pnl,
        "cum_alt_pnl_btc": total_alt_pnl_btc, "final_equity": equity,
        "btc_equiv": btc_equiv
    }
    return perf_df, summary


# --- MAIN ---
def main() -> None:
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Data loaded successfully from {CSV_PATH}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_PATH}")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    df["rebalance_ts"] = pd.to_datetime(df["rebalance_ts"])
    required_cols = ["rebalance_ts", "sym", "btc_price_usd", "price_usd", "price_btc", "mcap_btc", "rank"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in CSV: {', '.join(missing_cols)}")
        return

    key_cols_for_na_check = ["btc_price_usd", "price_btc", "mcap_btc", "rank"]
    initial_rows = len(df)
    df.dropna(subset=key_cols_for_na_check, inplace=True)
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f"Warning: Removed {rows_removed} rows with NaN values in key columns.")

    if df.empty:
        print("Error: DataFrame is empty after loading or cleaning. Cannot proceed.")
        return

    logger = PortfolioLogger()
    perf, summary = backtest_rank_altbtc_short(df, logger=logger, top_n=TOP_N)  # Ujisti se, že předáváš TOP_N

    if logger.log:
        try:
            logger.dump_csv(SNAP_CSV)
        except Exception as e:
            print(f"Error saving snapshots: {e}")
    elif not perf.empty:
        print("Logger recorded no data. Snapshots not saved.")

    if summary:
        print("\n--- Summary ---")
        print(f"Cumulative BTC P/L : {summary['cum_btc_pnl']:+,.2f} USD")
        print(f"Cumulative ALT P/L : {summary['cum_alt_pnl']:+,.2f} USD ({summary['cum_alt_pnl_btc']:+.6f} BTC)")
        print(f"Final equity       : {summary['final_equity']:,.2f} USD")
        print(f"Final equivalent   : {summary['btc_equiv']:.6f} BTC\n")
    else:
        print("Backtest did not produce a summary.")

    if not perf.empty and 'Equity_USD' in perf.columns and not perf['Equity_USD'].isnull().all():
        plot_data = perf.dropna(subset=['Equity_USD', 'Date'])
        if not plot_data.empty:
            min_equity = plot_data["Equity_USD"].min()
            max_equity = plot_data["Equity_USD"].max()
            ymin = min(START_CAP, min_equity) * 0.9
            ymax = max(START_CAP, max_equity) * 1.1
            ymin = max(0, ymin)
            if ymin >= ymax: ymax = ymin * 1.2 if ymin > 0 else START_CAP * 1.1

            fmt = ScalarFormatter(useOffset=False)
            fmt.set_scientific(False)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(plot_data["Date"], plot_data["Equity_USD"], marker=".", linestyle='-', label="Total Equity (USD)")
            ax.set_title(
                f"{BTC_W * 100:.0f}% BTC Long vs {ALT_W * 100:.0f}% ALT/BTC Short (Top {TOP_N}) - Weekly Rebalance")
            ax.set_xlabel("Date")
            ax.set_ylabel("Equity (USD)")

            if ymax > ymin: ax.set_ylim(ymin, ymax)
            ax.grid(True, which='major', linestyle='--', linewidth='0.5', alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', linewidth='0.5', alpha=0.5)
            ax.minorticks_on()
            ax.yaxis.set_major_formatter(fmt)
            ax.ticklabel_format(style='plain', axis='y')
            ax.axhline(START_CAP, color='red', linestyle='--', linewidth=1, label=f'Start Capital (${START_CAP:,.0f})')

            if summary:
                final_pnl = summary['final_equity'] - START_CAP
                perc_pnl = (final_pnl / START_CAP) * 100 if START_CAP != 0 else 0
                plt.text(0.02, 0.02, f"Total PNL: ${final_pnl:,.2f} ({perc_pnl:.2f}%)", transform=ax.transAxes,
                         fontsize=9,
                         verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
            fig.autofmt_xdate()
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("No valid equity data for plotting.")
    elif perf.empty:
        print("No performance data generated, skipping plot.")
    else:
        print("Equity data missing or all NaN, skipping plot.")


if __name__ == "__main__":
    main()