import os
import io
import base64
import datetime as dt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request, send_file, flash, redirect, url_for
from analyzer import backtest_rank, CSV_PATH, SNAP_CSV, START_CAP, BTC_W, ALT_W, TOP_N, EXCLUDED
from fetcher import run_fetcher

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev')

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Backtest GUI</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    table th { text-align: center; }
    table td { text-align: right; }
  </style>
</head>
<body class="p-4">
  <div class="container">
    <h1>Backtest GUI</h1>
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for cat, msg in messages %}
          <div class="alert alert-{{ cat }}">{{ msg }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}
    <form method="post">
      <div class="mb-3">
        <label class="form-label">CSV path</label>
        <input class="form-control" name="csv_path" value="{{ csv_path }}">
      </div>
      <div class="mb-3">
        <label class="form-label">Start capital (USD)</label>
        <input type="number" step="any" class="form-control" name="start_cap" value="{{ start_cap }}">
      </div>
      <div class="mb-3">
        <label class="form-label">BTC Weight</label>
        <input type="number" step="any" class="form-control" name="btc_w" value="{{ btc_w }}">
      </div>
      <div class="mb-3">
        <label class="form-label">ALT Weight</label>
        <input type="number" step="any" class="form-control" name="alt_w" value="{{ alt_w }}">
      </div>
      <div class="mb-3">
        <label class="form-label">Top N alts</label>
        <input type="number" class="form-control" name="top_n" value="{{ top_n }}">
      </div>
      <div class="mb-3">
        <label class="form-label">Excluded symbols (comma separated)</label>
        <input class="form-control" name="excluded" value="{{ excluded }}">
      </div>
      <button type="submit" class="btn btn-primary">Run Backtest</button>
    </form>

    <p><a class="btn btn-link" data-bs-toggle="collapse" href="#betaFetch" role="button" aria-expanded="false" aria-controls="betaFetch">Beta: Fetch Data</a></p>
    <div class="collapse" id="betaFetch">
      <form method="post" action="/fetch">
        <button type="submit" class="btn btn-warning mb-3">Fetch Data</button>
      </form>
    </div>

    {% if result %}
      <hr>
      <h2>CSV Metadata</h2>
      <ul>
        <li>File path: {{ csv_path }}</li>
        <li>Last modified: {{ meta.modified }}</li>
        <li>Rows: {{ meta.n_rows }}</li>
        <li>Snapshots: {{ meta.n_snapshots }}</li>
        <li>Date range: {{ meta.date_min }} – {{ meta.date_max }}</li>
      </ul>

      <h2>Backtest Summary</h2>
      <ul>
        <li>Cumulative BTC P/L (USD): {{ summary.cum_btc_pnl }}</li>
        <li>Cumulative ALT P/L (USD): {{ summary.cum_alt_pnl }}</li>
        <li>Final Equity (USD): {{ summary.final_equity }}</li>
        <li>Final Equity (BTC): {{ summary.btc_equiv }}</li>
      </ul>

      <h2>Equity Curve</h2>
      <img src="data:image/png;base64,{{ plot_data }}" class="img-fluid mb-3">

      <h2>Performance Table</h2>
      {{ perf_table|safe }}
      <p><a href="{{ download_perf_url }}" class="btn btn-secondary mt-3">Download perf CSV</a></p>

      <h2>Portfolio Snapshots</h2>
      {% if snap_table %}
        {{ snap_table|safe }}
        <p><a href="/download_snapshots" class="btn btn-secondary mt-2">Download Snapshots CSV</a></p>
      {% else %}
        <p>No snapshot data available.</p>
      {% endif %}
    {% endif %}
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

def get_defaults():
    return {
        'csv_path': str(CSV_PATH),
        'start_cap': START_CAP,
        'btc_w': BTC_W,
        'alt_w': ALT_W,
        'top_n': TOP_N,
        'excluded': ','.join(EXCLUDED)
    }

@app.route("/", methods=["GET", "POST"])
def index():
    context = get_defaults()
    if request.method == "POST":
        # Parse inputs
        csv_path = request.form.get('csv_path', context['csv_path'])
        try:
            start_cap = float(request.form.get('start_cap', context['start_cap']))
            btc_w = float(request.form.get('btc_w', context['btc_w']))
            alt_w = float(request.form.get('alt_w', context['alt_w']))
            top_n = int(request.form.get('top_n', context['top_n']))
            excluded = request.form.get('excluded', context['excluded'])
            excluded_list = [s.strip() for s in excluded.split(',') if s.strip()]
        except ValueError as e:
            flash(f'Invalid input: {e}', 'danger')
            return render_template_string(HTML, **context)
        # Load weights CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            flash(f'Failed to load CSV: {e}', 'danger')
            return render_template_string(HTML, **context)
        if 'rebalance_ts' in df.columns:
            df['rebalance_ts'] = pd.to_datetime(df['rebalance_ts'])
        # Metadata
        try:
            stat = os.stat(csv_path)
            modified = dt.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            modified = 'N/A'
        meta = {
            'modified': modified,
            'n_rows': len(df),
            'n_snapshots': df['rebalance_ts'].nunique() if 'rebalance_ts' in df.columns else 'N/A',
            'date_min': df['rebalance_ts'].min().strftime('%Y-%m-%d') if 'rebalance_ts' in df.columns else 'N/A',
            'date_max': df['rebalance_ts'].max().strftime('%Y-%m-%d') if 'rebalance_ts' in df.columns else 'N/A'
        }
        # Run backtest
        perf_df, summary = backtest_rank(df, btc_w=btc_w, alt_w=alt_w, top_n=top_n, start_cap=start_cap)
        # Plot equity curve
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(perf_df['Date'], perf_df['Equity_USD'], marker='o')
        ax.set_title('Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity (USD)')
        ax.grid(True, alpha=0.4)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        # Performance table (format numbers to 2 decimal places)
        perf_table = perf_df.to_html(
            classes="table table-striped",
            index=False,
            float_format="%.2f"
        )
        # Portfolio snapshots table
        try:
            df_snap = pd.read_csv(SNAP_CSV)
            if 'Date' in df_snap.columns:
                df_snap['Date'] = pd.to_datetime(df_snap['Date']).dt.strftime('%Y-%m-%d')
            snap_table = df_snap.to_html(
                classes="table table-striped",
                index=False,
                float_format="%.2f"
            )
        except Exception as e:
            flash(f'Failed to load snapshots: {e}', 'warning')
            snap_table = None
        # Download URLs
        from urllib.parse import urlencode
        params = urlencode({
            'csv_path': csv_path,
            'start_cap': start_cap,
            'btc_w': btc_w,
            'alt_w': alt_w,
            'top_n': top_n,
            'excluded': excluded
        })
        download_perf_url = f"/download_perf?{params}"
        context.update({
            'result': True,
            'csv_path': csv_path,
            'start_cap': start_cap,
            'btc_w': btc_w,
            'alt_w': alt_w,
            'top_n': top_n,
            'excluded': excluded,
            'meta': meta,
            'summary': summary,
            'plot_data': plot_data,
            'perf_table': perf_table,
            'snap_table': snap_table,
            'download_perf_url': download_perf_url
        })
    return render_template_string(HTML, **context)

@app.route('/download_perf')
def download_perf():
    # Get parameters
    csv_path = request.args.get('csv_path', str(CSV_PATH))
    try:
        start_cap = float(request.args.get('start_cap', START_CAP))
        btc_w = float(request.args.get('btc_w', BTC_W))
        alt_w = float(request.args.get('alt_w', ALT_W))
        top_n = int(request.args.get('top_n', TOP_N))
        excluded = request.args.get('excluded', ','.join(EXCLUDED))
        excluded_list = [s.strip() for s in excluded.split(',') if s.strip()]
    except ValueError as e:
        flash(f'Invalid input: {e}', 'danger')
        return redirect(url_for('index'))
    # Load and run backtest
    df = pd.read_csv(csv_path)
    if 'rebalance_ts' in df.columns:
        df['rebalance_ts'] = pd.to_datetime(df['rebalance_ts'])
    perf_df, _ = backtest_rank(df, btc_w=btc_w, alt_w=alt_w, top_n=top_n, start_cap=start_cap)
    buf = io.StringIO()
    perf_df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.read().encode('utf-8')),
        mimetype='text/csv',
        download_name='perf.csv',
        as_attachment=True
    )
@app.route('/download_snapshots')
def download_snapshots():
    # Serve the portfolio_snapshots.csv file
    try:
        return send_file(str(SNAP_CSV), as_attachment=True)
    except Exception as e:
        flash(f'Could not download snapshots: {e}', 'danger')
        return redirect(url_for('index'))

@app.route('/fetch', methods=['POST'])
def fetch():
    try:
        run_fetcher()
        flash('Data fetched successfully', 'success')
    except Exception as e:
        flash(f'Fetch failed: {e}', 'danger')
    return redirect(url_for('index'))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)