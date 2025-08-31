"""Utilities for printing readable, consistent console outputs.

- Uses ASCII-safe tables via tabulate (no ANSI colors by default),
  so output stays readable in terminals and inside Streamlit's st.text.
- Automatically adapts section width to terminal size.
- Provides helpers for common number formatting.

Set PLAIN_OUTPUT=1 to force simple separators (useful for very constrained environments).
Detects basic Streamlit env and avoids any ANSI/coloration by default.
"""
from __future__ import annotations

import os
import shutil
from typing import Iterable, Mapping, Any, Sequence, Optional

from tabulate import tabulate


def _term_width(default: int = 100) -> int:
    try:
        # Respect COLUMNS if set, else ask terminal, else fallback
        if (cols := os.environ.get("COLUMNS")):
            return max(int(cols), 40)
        return max(shutil.get_terminal_size((default, 24)).columns, 40)
    except Exception:
        return default


def _is_plain() -> bool:
    # Prefer plain output in Streamlit or when explicitly requested
    if os.environ.get("PLAIN_OUTPUT"):
        return True
    # Streamlit commonly sets these
    if os.environ.get("STREAMLIT_SERVER_ENABLED") or os.environ.get("STREAMLIT_RUNTIME"):
        return True
    return False


def section(title: str, width: Optional[int] = None) -> None:
    """Print a prominent section header with consistent width."""
    w = width or min(_term_width(), 120)
    sep = "-" * w if _is_plain() else "═" * w
    print("\n" + sep)
    centered = f" {title.strip()} "
    if _is_plain():
        print(centered.center(w, " "))
    else:
        # Box style top/bottom borders for better readability in terminals
        print(centered.center(w, " "))
    print(sep)


def kv_table(data: Mapping[str, Any], title: Optional[str] = None, width: Optional[int] = None) -> None:
    """Print a two-column key/value table."""
    if title:
        section(title, width)
    rows = [[str(k), _fmt_any(v)] for k, v in data.items()]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="grid", stralign="left", numalign="right"))


def table(rows: Iterable[Sequence[Any]], headers: Sequence[str], title: Optional[str] = None, width: Optional[int] = None) -> None:
    """Print a generic table with grid borders."""
    if title:
        section(title, width)
    safe_rows = [[_fmt_any(x) for x in r] for r in rows]
    print(tabulate(safe_rows, headers=list(headers), tablefmt="grid", stralign="right", numalign="right"))


def _fmt_any(v: Any) -> str:
    if isinstance(v, float):
        # Keep decent precision for floats; caller can pre-format
        return f"{v:,.4f}" if abs(v) < 1 else f"{v:,.2f}"
    if isinstance(v, (int,)):
        return f"{v:,}"
    return str(v)


def fmt_money(v: float, currency: str = "$") -> str:
    return f"{currency}{v:,.2f}"


def fmt_pct(v: float, places: int = 2, sign: bool = False) -> str:
    if sign:
        return f"{v:+.{places}f}%"
    return f"{v:.{places}f}%"


def fmt_qty(v: float, places: int = 6) -> str:
    return f"{v:.{places}f}"

