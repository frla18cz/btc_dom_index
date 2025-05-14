import pandas as pd, duckdb, pyarrow as pa, pyarrow.parquet as pq

# 1) stáhnu ručně nebo přes Kaggle CLI: kaggle datasets download bizzyvinci/coinmarketcap-historical-data
df = pd.read_csv("coinmarketcap_2013_to_2025.csv", parse_dates=["Date"])

# 2) přesun do DuckDB kvůli rychlosti
con = duckdb.connect()
con.execute("CREATE TABLE cmc AS SELECT * FROM df")

# 3) týdenní snapshot – každé pondělí
weekly = con.execute("""
    WITH w AS (
        SELECT *,
               date_trunc('week', date) + INTERVAL '1 day' AS monday -- week starting Monday
        FROM cmc
    )
    SELECT monday AS snapshot_date,
           symbol,
           MAX(market_cap) AS mcap
    FROM w
    GROUP BY snapshot_date, symbol
""").fetch_df()

top50 = (weekly.sort_values(["snapshot_date", "mcap"], ascending=[True, False])
                .groupby("snapshot_date")
                .head(50))

pq.write_table(pa.Table.from_pandas(top50), "top50_weekly.parquet")
