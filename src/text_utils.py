"""
text_utils.py

Pure, stateless text-processing helpers.

Functions:
    table_html_to_text   — Convert an HTML table to a readable plain-text string.
    table_column_headers — Extract only the column header names from an HTML table.
    clean                — Collapse excessive whitespace in a string.

These utilities have no dependencies on project-specific config so they can be
unit-tested in isolation.
"""

from __future__ import annotations

import re
from io import StringIO

import pandas as pd


def table_html_to_text(html: str) -> str:
    """Convert an HTML table to a plain-text string via pandas.

    Args:
        html: Raw HTML string containing a ``<table>`` element.

    Returns:
        A whitespace-aligned text representation of the first table found, or
        an empty string if parsing fails or no table is present.
    """
    try:
        dfs = pd.read_html(StringIO(html))
        if not dfs:
            return ""
        return dfs[0].to_string(index=False)
    except Exception:
        return ""


def table_column_headers(html: str) -> str:
    """Extract only the column header names from an HTML table.

    Useful for building a compact embedding text that captures table structure
    without reproducing all the cell data.

    Args:
        html: Raw HTML string containing a ``<table>`` element.

    Returns:
        Pipe-delimited column header names, or an empty string on failure.

    Example:
        >>> table_column_headers("<table><tr><th>Name</th><th>Value</th></tr></table>")
        'Name | Value'
    """
    try:
        dfs = pd.read_html(StringIO(html))
        if not dfs:
            return ""
        cols = dfs[0].columns.tolist()
        return " | ".join(str(c) for c in cols)
    except Exception:
        return ""


def clean(text: str) -> str:
    """Collapse runs of whitespace (spaces, tabs, newlines) into a single space.

    Args:
        text: Input string that may contain excessive whitespace.

    Returns:
        Stripped string with all internal whitespace normalised to single spaces.

    Example:
        >>> clean("  hello\\n  world  ")
        'hello world'
    """
    return re.sub(r"\s+", " ", text).strip()
