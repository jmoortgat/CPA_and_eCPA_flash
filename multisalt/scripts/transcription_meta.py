#!/usr/bin/env python3
"""Source line and true temperature of a raw transcription file.

Every transcription carries its temperature in the name of the directory
holding it (``T323K/``, ``298K/``). That is lossy, and in two ways that
matter:

* it cannot represent a measured isotherm that is not an integer kelvin,
  so 351.65 K (Bastami 2014) and 375.15 K become 353 K and 373 K; and
* the same integer means different things to different sources. 343 K is
  a genuine 70 degC isotherm for Blanco 1977 and Susak 1980, and is
  344.26 K -- 160 degF -- for Culberson 1951 and Olds 1942, who report on
  a Fahrenheit grid. One directory, two temperatures.

A file may therefore declare its own temperature on the source line,
after a pipe:

    #CULBERSON(1951) | T=344.26 | 160 degF, Table I

Everything before the pipe is the source key, exactly as before. The
``T=`` field, when present, is the temperature the *paper* reports and
wins over the directory. Anything after it is a free-text justification
and is ignored by the parsers -- it exists so the claim can be audited
against the paper.

Files without a pipe behave exactly as they always did, so this is
backwards compatible and can be applied one verified file at a time.
"""
from __future__ import annotations

import re
from pathlib import Path

_T_FIELD = re.compile(r"\bT\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*K?\b", re.I)


def parse_source_line(line: str) -> tuple[str, float | None, str]:
    """Split a transcription's first line into (source, T_K|None, note)."""
    body = line.lstrip("#").strip()
    if "|" not in body:
        return body, None, ""
    parts = [p.strip() for p in body.split("|")]
    source = parts[0]
    T = None
    note_parts = []
    for p in parts[1:]:
        m = _T_FIELD.fullmatch(p) or _T_FIELD.match(p)
        if m and T is None and p.lower().lstrip().startswith("t"):
            T = float(m.group(1))
        else:
            note_parts.append(p)
    return source, T, " | ".join(note_parts)


def temperature_from_dirname(name: str) -> float | None:
    """`T323K` or `298K` -> 323.0 / 298.0; None if it is not an isotherm dir."""
    m = re.fullmatch(r"T?(\d+)K", name)
    return float(m.group(1)) if m else None


def read_header(path: Path) -> tuple[str, float | None, str]:
    """(source, declared T_K or None, note) for a transcription file."""
    with open(path, errors="replace") as fh:
        return parse_source_line(fh.readline())


def true_temperature(path: Path, fallback: float | None = None) -> float | None:
    """Temperature to use: the file's own declaration, else the directory."""
    _, T, _ = read_header(Path(path))
    if T is not None:
        return T
    if fallback is not None:
        return float(fallback)
    return temperature_from_dirname(Path(path).parent.name)
