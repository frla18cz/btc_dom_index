import sqlite3
import pandas as pd

class Storage:
    """
    Storage class for persisting token snapshots and token metadata in SQLite.
    Creates two tables:
      - tokens(sym TEXT PRIMARY KEY, first_seen TEXT, last_seen TEXT)
      - snapshots(sym TEXT, week_ts TEXT, rank INTEGER,
                  mcap_btc REAL, price_btc REAL, price_usd REAL,
                  btc_price_usd REAL, weight REAL,
                  PRIMARY KEY(sym, week_ts))
    """
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.init_tables()

    def init_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                sym TEXT PRIMARY KEY,
                first_seen TEXT,
                last_seen TEXT
            );
        """ )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                sym TEXT,
                week_ts TEXT,
                rank INTEGER,
                mcap_btc REAL,
                price_btc REAL,
                price_usd REAL,
                btc_price_usd REAL,
                weight REAL,
                PRIMARY KEY(sym, week_ts)
            );
        """ )
        self.conn.commit()

    def store_snapshot(self, df: pd.DataFrame):
        """
        Store snapshot DataFrame into the snapshots and tokens tables.
        """
        cursor = self.conn.cursor()
        for _, row in df.iterrows():
            sym = row['sym']
            week_ts = row['rebalance_ts'].isoformat() if hasattr(row['rebalance_ts'], 'isoformat') else str(row['rebalance_ts'])
            rank = int(row.get('rank', 0))
            mcap_btc = float(row.get('mcap_btc', 0.0))
            price_btc = float(row.get('price_btc', 0.0))
            price_usd = float(row.get('price_usd')) if row.get('price_usd') is not None else None
            btc_price_usd = float(row.get('btc_price_usd')) if row.get('btc_price_usd') is not None else None
            weight = float(row.get('weight', 0.0))
            # Insert or ignore into tokens table
            cursor.execute(
                "INSERT OR IGNORE INTO tokens (sym, first_seen, last_seen) VALUES (?, ?, ?)",
                (sym, week_ts, week_ts)
            )
            # Update last_seen if this snapshot is newer
            cursor.execute(
                "UPDATE tokens SET last_seen = ? WHERE sym = ? AND last_seen < ?",
                (week_ts, sym, week_ts)
            )
            # Insert or replace snapshot record
            cursor.execute(
                """
                INSERT OR REPLACE INTO snapshots
                  (sym, week_ts, rank, mcap_btc, price_btc, price_usd, btc_price_usd, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (sym, week_ts, rank, mcap_btc, price_btc, price_usd, btc_price_usd, weight)
            )
        self.conn.commit()

    def close(self):
        self.conn.close()