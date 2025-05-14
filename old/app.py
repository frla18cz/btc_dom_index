import os
import pandas as pd
from flask import Flask, render_template_string, send_file, request
from fetcher import run_fetcher
from config.config import OUTPUT_FILENAME

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BTC Dominance GUI</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="p-4">
  <div class="container">
    <h1>BTC Dominance – Top Altcoin Weights</h1>
    <form method="post">
      <button type="submit" class="btn btn-primary my-3">Fetch & Refresh</button>
    </form>
    {% if table %}
      <h2>Výsledky</h2>
      {{ table|safe }}
      <p><a href="/download" class="btn btn-secondary mt-2">Stáhnout CSV</a></p>
    {% endif %}
  </div>
</body>
</html>
"""

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    table = None
    if request.method == "POST":
        df = run_fetcher()
        table = df.to_html(classes="table table-striped", index=False)
    return render_template_string(HTML, table=table)

@app.route("/download")
def download():
    return send_file(OUTPUT_FILENAME, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)