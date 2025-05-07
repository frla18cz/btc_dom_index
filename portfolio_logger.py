# portfolio_logger.py
from pathlib import Path
import pandas as pd
from typing import Dict, List

class PortfolioLogger:
    """
    Jednoduchý logger pro back-test portfolia.
    Uchovává snapshoty pozic a dovolí export do CSV.
    """
    def __init__(self) -> None:
        self.records: List[dict] = []

    # ------------------------------------------------------------------ #
    def record(
        self,
        date,                       # pd.Timestamp
        equity_usd: float,
        btc_qty: float,
        alt_qty: Dict[str, float],  # {sym: qty (negativní = short)}
        btc_price: float,
        alt_prices: Dict[str, float],
        btc_pnl: float,
        alt_pnl: float,
    ) -> None:
        """Uloží snapshot a základní P/L statistiky."""
        gross_long  = abs(btc_qty * btc_price)
        gross_short = sum(abs(qty) * alt_prices[s] for s, qty in alt_qty.items())
        gross_exp   = gross_long + gross_short
        net_exp     = gross_long - gross_short

        self.records.append({
            "Date": date,
            "Equity_USD": equity_usd,
            "Gross_Long_USD": gross_long,
            "Gross_Short_USD": gross_short,
            "Gross_Exposure_USD": gross_exp,
            "Net_Exposure_USD": net_exp,
            "BtcQty": btc_qty,
            "AltCount": len(alt_qty),
            "Btc_PnL": btc_pnl,
            "Alts_PnL": alt_pnl,
            **{f"{sym}_qty": qty for sym, qty in alt_qty.items()},
        })

    # ------------------------------------------------------------------ #
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    # ------------------------------------------------------------------ #
    def dump_csv(self, path: Path | str = "portfolio_snapshots.csv") -> None:
        self.to_dataframe().to_csv(path, index=False)
        print(f"✓ Portfolio snapshots uložen do {path!s}")
