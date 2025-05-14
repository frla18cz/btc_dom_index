# portfolio_logger.py
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional # Přidáno Optional

class PortfolioLogger:
    """
    Jednoduchý logger pro back-test portfolia.
    Uchovává snapshoty pozic a dovolí export do CSV.
    """
    def __init__(self) -> None:
        self.log: List[dict] = [] # Přejmenováno z records na log pro konzistenci s dump_csv

    # ------------------------------------------------------------------ #
    def record(
        self,
        date,                       # pd.Timestamp
        equity_usd: float,
        btc_qty: float,
        alt_qty: Dict[str, float],  # {sym: qty (negativní = short)}
        btc_price: float, # Předpokládá BTC/USD
        alt_prices: Dict[str, float | None], # Předpokládá ALT/USD
        btc_pnl: float,
        alt_pnl: float, # Předpokládá ALT PNL v USD
    ) -> None:
        """Původní metoda - uloží snapshot a základní P/L statistiky (ALT/USD focus)."""
        gross_long_usd = abs(btc_qty * btc_price) if not pd.isna(btc_qty) and not pd.isna(btc_price) else 0

        gross_short_usd = 0
        valid_alt_count = 0
        alt_qtys_log = {}

        for s, qty in alt_qty.items():
            price_usd = alt_prices.get(s)
            if price_usd is not None and not pd.isna(price_usd) and not pd.isna(qty):
                gross_short_usd += abs(qty * price_usd)
                valid_alt_count += 1
            alt_qtys_log[f"{s}_qty"] = qty # Logujeme množství i když cena chybí

        gross_exp_usd = gross_long_usd + gross_short_usd
        net_exp_usd = gross_long_usd - gross_short_usd

        self.log.append({ # Opraveno na self.log
            "Date": date,
            "Equity_USD": equity_usd,
            "Gross_Long_USD": gross_long_usd,
            "Gross_Short_USD": gross_short_usd, # Hodnota založená na ALT/USD
            "Gross_Exposure_USD": gross_exp_usd,
            "Net_Exposure_USD": net_exp_usd,
            "BtcQty": btc_qty,
            "AltCount": len(alt_qty),
            "ValidAltUsdValueCount": valid_alt_count,
            "Btc_PnL_USD": btc_pnl,
            "Alts_PnL_USD": alt_pnl,
            **alt_qtys_log,
        })

    # --- NOVÁ METODA PRO ALT/BTC SHORT ---
    def record_altbtc(
        self,
        date,                       # pd.Timestamp
        equity_usd: float,
        btc_qty: float,
        alt_qty: Dict[str, float],  # {sym: qty (negativní = short)}
        btc_price_usd: float,       # BTC/USD cena
        alt_prices_usd: Dict[str, Optional[float]], # ALT/USD ceny (pro info)
        alt_prices_btc: Dict[str, Optional[float]], # ALT/BTC ceny (klíčové)
        btc_pnl_usd: float,         # Týdenní BTC PNL v USD
        alt_pnl_usd: float,         # Týdenní ALT PNL v USD (přepočteno z BTC)
        alt_pnl_btc: float,         # Týdenní ALT PNL v BTC
    ) -> None:
        """Uloží snapshot pro strategii short ALT/BTC."""
        gross_long_usd = abs(btc_qty * btc_price_usd) if not pd.isna(btc_qty) and not pd.isna(btc_price_usd) else 0

        # Výpočet skutečné USD hodnoty short legu přes ALT/BTC ceny
        gross_short_usd_altbtc = 0
        valid_alt_count = 0
        alt_qtys_log = {}

        for s, qty in alt_qty.items():
            price_btc = alt_prices_btc.get(s) # Získáme ALT/BTC cenu
            # Ověříme, zda máme platné ceny pro výpočet hodnoty
            if price_btc is not None and not pd.isna(price_btc) and btc_price_usd is not None and not pd.isna(btc_price_usd) and qty is not None and not pd.isna(qty):
                 short_val_usd = abs(qty * price_btc * btc_price_usd)
                 gross_short_usd_altbtc += short_val_usd
                 valid_alt_count += 1
            alt_qtys_log[f"{s}_qty"] = qty # Logujeme množství vždy

        gross_exp_usd = gross_long_usd + gross_short_usd_altbtc
        net_exp_usd = gross_long_usd - gross_short_usd_altbtc # Expozice je Long BTC vs Short ALT

        self.log.append({ # Opraveno na self.log
            "Date": date,
            "Equity_USD": equity_usd,
            "Gross_Long_USD": gross_long_usd,
            "Gross_Short_USD_AltBTC": gross_short_usd_altbtc, # USD hodnota short legu
            "Gross_Exposure_USD": gross_exp_usd,
            "Net_Exposure_USD": net_exp_usd,
            "BtcQty": btc_qty,
            "AltCount": len(alt_qty), # Počet shortovaných altů
            "ValidAltBtcValueCount": valid_alt_count, # Počet altů, u kterých šla spočítat hodnota
            "Btc_PnL_USD": btc_pnl_usd,
            "Alts_PnL_USD": alt_pnl_usd,
            "Alts_PnL_BTC": alt_pnl_btc, # Přidáno PNL altů v BTC
            **alt_qtys_log, # Jednotlivá množství altů
        })
    # ------------------------------------------------------------------ #

    @property # Uděláme z toho property pro snadnější přístup, pokud je potřeba
    def records(self) -> List[dict]:
         # Zajistí zpětnou kompatibilitu, pokud by nějaký kód stále používal self.records
         return self.log

    def to_dataframe(self) -> pd.DataFrame:
        # Zkontrolujeme, zda log není prázdný
        if not self.log:
            return pd.DataFrame() # Vrátíme prázdný DataFrame
        return pd.DataFrame(self.log)

    # ------------------------------------------------------------------ #
    def dump_csv(self, path: Path | str = "portfolio_snapshots.csv") -> None:
        df_to_save = self.to_dataframe()
        if not df_to_save.empty:
            df_to_save.to_csv(path, index=False, float_format='%.8f') # Přidáno formátování pro floaty
            print(f"✓ Portfolio snapshots saved to {path!s}")
        else:
            print("! No records to save to CSV.")