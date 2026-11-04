"""Helpers for the executable paper notebooks in ../notebooks/.

    from nbtools import run, show, latex_table, show_table

``run(script)`` runs a figure or table script of this package once per session,
from the package directory, with the package on PYTHONPATH. With
``RUN_LONG = True`` it passes ``--force`` so the script recomputes its cached
model results instead of plotting from ``results_cache/``.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]          # multisalt/
FIG = Path(os.environ.get("ECPA_PAPER_FIGURES", PKG / "figures"))
TAB = Path(os.environ.get("ECPA_PAPER_DIR", PKG / "paper_tables"))

RUN_LONG = False
_done: set = set()


def run(script: str, *args: str, long: tuple | None = None, cwd: str | None = None,
        module: bool = False) -> None:
    """Run a script of the package (path relative to multisalt/) once per session.

    args are used by default; with RUN_LONG the arguments are long, which
    defaults to args + ("--force",) for the scripts in scripts/ (recompute the
    cached model results) and to args otherwise.
    """
    if RUN_LONG:
        if long is None:
            long = tuple(args) + (("--force",) if script.startswith("scripts/") else ())
        args = tuple(long)
    key = (script, tuple(args))
    if key in _done:
        return
    cmd = [sys.executable] + (["-m", script] if module else [str(PKG / script)])
    cmd += list(args)
    env = {**os.environ, "PYTHONPATH": str(PKG) + os.pathsep + str(PKG / "scripts"),
           "MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore"}
    r = subprocess.run(cmd, cwd=cwd or PKG, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print((r.stdout or "")[-1500:])
        print((r.stderr or "")[-3000:])
        raise RuntimeError(f"{script} failed")
    _done.add(key)


def show(*paths, width: int = 820, max_px: int = 1640) -> None:
    """Display PNG files (paths relative to multisalt/figures/ unless absolute).

    The image embedded in the notebook is reduced to at most ``max_px`` pixels
    wide (twice the display width, for high-resolution screens) and 256
    colours; the file on disk keeps its full resolution.
    """
    import io
    from IPython.display import Image, display
    for p in paths:
        p = Path(p)
        if not p.is_absolute():
            p = (FIG / p) if (FIG / p).exists() else (PKG / p)
        if not p.exists():
            print("not found:", p)
            continue
        data = p.read_bytes()
        try:
            from PIL import Image as PILImage
            im = PILImage.open(io.BytesIO(data)).convert("RGB")
            if im.width > max_px:
                im = im.resize((max_px, round(im.height * max_px / im.width)), PILImage.LANCZOS)
            # 256 colours are enough for line plots and keep the notebook small
            im = im.quantize(colors=256, method=PILImage.Quantize.MEDIANCUT,
                             dither=PILImage.Dither.NONE)
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            data = buf.getvalue()
        except ImportError:
            pass
        display(Image(data=data, format="png", width=width))


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

_SUB = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")
_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


def _ce(x: str) -> str:
    x = re.sub(r"\^\{?(\d*[+\-])\}?", lambda m: m.group(1).translate(_SUP), x)
    return re.sub(r"(?<=[A-Za-z)])(\d+)", lambda m: m.group(1).translate(_SUB), x)


def _cell(c: str) -> str:
    c = c.strip()
    c = re.sub(r"\\(textcite|citet|cite)\{([^}]*)\}", r"\2", c)
    c = re.sub(r"\\textbf\{([^}]*)\}", r"\1", c)
    c = re.sub(r"\\ce\{([^}]*)\}", lambda m: _ce(m.group(1)), c)
    c = re.sub(r"\\(textit|emph)\{([^}]*)\}", r"\2", c)
    c = c.replace("\\,", " ").replace("---", "—").replace("--", "–").replace("{", "").replace("}", "").replace("$", "").replace("\\%", "%").replace("~", " ")
    return re.sub(r"\\[a-zA-Z]+", "", c).strip()


def latex_table(path, ncol: int):
    """Rows of a generated LaTeX tabular as a DataFrame (section rows kept as 'group')."""
    import pandas as pd
    t = Path(path).read_text()
    rows, group, header = [], "", None
    for ln in t.splitlines():
        ln = ln.strip()
        if ln.startswith("%"):
            continue
        m = re.match(r"\\multicolumn\{\d+\}\{[lcr]\}\{(.*)\}\s*\\\\", ln)
        if m and "continued" not in m.group(1).lower():
            group = _cell(m.group(1))
            continue
        if "&" not in ln or "\\\\" not in ln:
            continue
        cells = [_cell(c) for c in ln.split("\\\\")[0].split("&")]
        if len(cells) != ncol:
            continue
        if header is None and ("Reference" in cells[0] or "Salt" in cells[0]):
            header = cells
            continue
        if cells == header:
            continue
        rows.append([group] + cells)
    cols = ["group"] + (header or [f"c{i}" for i in range(ncol)])
    return pd.DataFrame(rows, columns=cols)


def show_table(df, hide_group: bool = False) -> None:
    """Display a DataFrame as a compact table (group rows as section headers).

    A column named by a tuple (heading, subheading) is shown under a two-row
    header in which consecutive columns with the same heading share one cell.
    """
    from IPython.display import HTML, display
    td = 'style="padding:2px 8px;text-align:left;border:none"'
    th = 'style="padding:2px 8px;text-align:left;border:none;border-bottom:1px solid"'
    thc = th.replace("text-align:left", "text-align:center")
    cols = [c for c in df.columns if c != "group"]
    out = ['<table style="border-collapse:collapse;border-top:2px solid;border-bottom:2px solid;'
           'font-size:90%;width:auto">']
    if any(isinstance(c, tuple) for c in cols):
        top, i = [], 0
        while i < len(cols):
            c = cols[i]
            if not isinstance(c, tuple):
                top.append(f'<th rowspan="2" {th}>{c}</th>')
                i += 1
                continue
            n = 1
            while i + n < len(cols) and isinstance(cols[i + n], tuple) and cols[i + n][0] == c[0]:
                n += 1
            top.append(f'<th colspan="{n}" {thc}>{c[0]}</th>')
            i += n
        out.append("<tr>" + "".join(top) + "</tr>")
        out.append("<tr>" + "".join(f"<th {th}>{c[1]}</th>" for c in cols if isinstance(c, tuple)) + "</tr>")
    else:
        out.append("<tr>" + "".join(f"<th {th}>{c}</th>" for c in cols) + "</tr>")
    last = None
    for _, r in df.iterrows():
        g = r.get("group", "")
        if not hide_group and g and g != last:
            out.append(f'<tr><td colspan="{len(cols)}" {td}><i>{g}</i></td></tr>')
            last = g
        bold = str(r[cols[0]]).startswith("Global") or str(r[cols[0]]).startswith("Total")
        cells = "".join(f"<td {td}>{'<b>' if bold else ''}{r[c]}{'</b>' if bold else ''}</td>" for c in cols)
        out.append(f"<tr>{cells}</tr>")
    out.append("</table>")
    display(HTML("".join(out)))


def side_by_side(published, released, on, cols):
    """One table from the published and the released version of a table.

    Rows are matched on the columns ``on``. A cell holds the single value
    where the two versions agree; where they differ it holds the released value
    followed by the published one in parentheses, e.g. ``0.17–6 (published:
    1–6)``. Rows found in only one version are marked as such. Published rows
    come first, in their published order, followed by rows found only in the
    released version.
    """
    import pandas as pd
    p = published.set_index(on)
    r = released.set_index(on)
    keys = list(p.index) + [k for k in r.index if k not in p.index]
    rows = []
    for k in keys:
        row = dict(zip(on, k if isinstance(k, tuple) else (k,)))
        for c in cols:
            a = p[c].get(k, "–") if k in p.index else "–"
            b = r[c].get(k, "–") if k in r.index else "–"
            if a == b:
                row[c] = a
            elif k not in p.index:
                row[c] = f"{b} (not in the published table)"
            elif k not in r.index:
                row[c] = f"– (published: {a})"
            else:
                row[c] = f"{b} (published: {a})"
        rows.append(row)
    return pd.DataFrame(rows)
