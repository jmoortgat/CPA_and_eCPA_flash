"""Locate, and if necessary fetch, the GasBrineBench experimental database.

Every figure, table and fitted parameter of Papers II and III is computed
against GasBrineBench. All code in this package that reads the database finds
it through this module. A missing database is fetched; it is never replaced by
other data.

Resolution order
----------------
1. ``ECPA_GBB``: an explicit path. It takes precedence and is never fetched or
   checked for updates.
2. An existing sibling checkout, found by walking up from this file.
3. The local cache (``ECPA_GBB_CACHE``, else ``~/.cache/ecpa/GasBrineBench``).
4. A fetch into the cache.

Updates
-------
``ECPA_GBB_REFRESH`` selects the policy:

``check`` (default)
    Ask the remote for its current commit (``git ls-remote``, about a second,
    no download) and fetch again only when it differs from the copy on disk.
``always``
    Fetch every time.
``never``
    Use the copy on disk, for offline runs and for runs that must use the
    pinned revision only.

A network failure is not fatal: if a check or a fetch fails and a usable copy
exists, that copy is used and the reason is reported. An error is raised only
when there is no local copy and the fetch fails.

Reproducibility
---------------
``PINNED_REF`` is the revision used for the papers. Fetches request it by
default; ``ECPA_GBB_REF`` overrides it (``main`` for the latest revision).
``describe`` reports the path, the revision and how the copy was obtained, so
that a figure can record the database it was computed from.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

#: Read-only HTTPS; public, so no credentials or SSH keys are needed.
REPO_URL = "https://github.com/jmoortgat/GasBrineBench.git"

#: The revision production runs get. Bump deliberately, with a re-run.
#:
#: A tag, not a branch: the papers cite a database version, so the default has
#: to name a fixed row set rather than whatever the tip happens to hold.
#: ``v1.1.1`` (11,537 data points, 111 sources, doi:10.5281/zenodo.22898153) is
#: the release behind Papers II and III: v1.1.0 with three mixed-brine recipes
#: corrected.
PINNED_REF = "v1.1.1"

#: The exact commit the reported numbers were produced against. ``describe``
#: says so when the resolved checkout sits somewhere else, so a figure built on
#: a different database cannot pass unnoticed.
PINNED_COMMIT = "b8d1f81b459b378f3b2fdf58078a5ce112137fd8"

#: The only two directories anything here needs: the CSVs and the loader
#: package. Together about 1.5 MB, against 21 MB for the whole repository
#: (the bulk is transcriptions and notebooks, which nothing imports).
SPARSE_PATHS = ("data", "gasbrinebench")

#: The file whose presence means "this really is a GasBrineBench checkout".
SENTINEL = Path("data") / "solubility.csv"

_RESOLVED: Path | None = None
_HOW: str = "not resolved yet"


def _run(args, cwd=None, timeout=120):
    return subprocess.run(args, cwd=cwd, timeout=timeout,
                          capture_output=True, text=True)


def cache_dir() -> Path:
    """Return the directory of the local GasBrineBench cache.

    Returns
    -------
    pathlib.Path
        ``ECPA_GBB_CACHE`` (with ``~`` expanded) when that environment
        variable is set, else ``~/.cache/ecpa/GasBrineBench``.  The directory
        need not exist.
    """
    env = os.environ.get("ECPA_GBB_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "ecpa" / "GasBrineBench"


def is_checkout(path: Path) -> bool:
    """Return True if ``path`` holds a GasBrineBench checkout.

    Parameters
    ----------
    path : pathlib.Path
        Directory to test.

    Returns
    -------
    bool
        True when ``path / SENTINEL`` (``data/solubility.csv``) exists.
    """
    return (path / SENTINEL).exists()


def find_local() -> Path | None:
    """An existing checkout, or None. Never touches the network."""
    env = os.environ.get("ECPA_GBB")
    if env:
        p = Path(env).expanduser()
        return p if is_checkout(p) else None
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "GasBrineBench"
        if is_checkout(cand):
            return cand
    c = cache_dir()
    return c if is_checkout(c) else None


def local_rev(root: Path) -> str | None:
    """Return the commit hash checked out in a local GasBrineBench directory.

    Runs ``git rev-parse HEAD`` in ``root``; no network access.

    Parameters
    ----------
    root : pathlib.Path
        The checkout, for example the result of ``find_local``.

    Returns
    -------
    str or None
        The 40-character commit hash, or None if ``root`` is not a git
        repository or git reports an error.
    """
    r = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    return r.stdout.strip() if r.returncode == 0 else None


def remote_rev(ref: str) -> str | None:
    """Commit the remote currently has for ``ref``. None if unreachable."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        r = subprocess.run(["git", "ls-remote", REPO_URL, ref, f"{ref}^{{}}"],
                           capture_output=True, text=True, timeout=30, env=env)
    except Exception:
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    lines = [ln.split("\t")[0] for ln in r.stdout.strip().splitlines()]
    # A tag resolves to two lines; the dereferenced one (^{}) is the commit.
    return lines[-1]


def acquire(dest: Path, ref: str | None = None) -> Path:
    """Sparse, shallow clone of the data and loader into ``dest``.

    Written to a scratch directory and moved into place, so an interrupted
    fetch cannot leave a half-populated tree that ``is_checkout`` would
    then accept.
    """
    ref = ref or os.environ.get("ECPA_GBB_REF") or PINNED_REF
    dest = Path(dest).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    scratch = dest.with_name(dest.name + ".fetching")
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)

    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    is_sha = len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower())
    if is_sha:
        # --branch does not take a commit, so init and fetch that one object.
        scratch.mkdir(parents=True)
        steps = [["git", "-C", str(scratch), "init", "-q"],
                 ["git", "-C", str(scratch), "remote", "add", "origin", REPO_URL],
                 ["git", "-C", str(scratch), "sparse-checkout", "init", "--cone"],
                 ["git", "-C", str(scratch), "sparse-checkout", "set", *SPARSE_PATHS],
                 ["git", "-C", str(scratch), "fetch", "--depth", "1",
                  "--filter=blob:none", "origin", ref],
                 ["git", "-C", str(scratch), "checkout", "-q", "FETCH_HEAD"]]
        for step in steps:
            r = subprocess.run(step, capture_output=True, text=True,
                               timeout=600, env=env)
            if r.returncode != 0:
                shutil.rmtree(scratch, ignore_errors=True)
                raise RuntimeError(
                    f"could not fetch GasBrineBench {ref} from {REPO_URL}:\n"
                    f"{r.stderr.strip()[:600]}")
    else:
        clone = ["git", "clone", "--depth", "1", "--filter=blob:none",
                 "--sparse", "--branch", ref, REPO_URL, str(scratch)]
        r = subprocess.run(clone, capture_output=True, text=True,
                           timeout=600, env=env)
        if r.returncode != 0:
            shutil.rmtree(scratch, ignore_errors=True)
            raise RuntimeError(
                f"could not fetch GasBrineBench {ref} from {REPO_URL}:\n"
                f"{r.stderr.strip()[:600]}")
        r = _run(["git", "-C", str(scratch), "sparse-checkout", "set",
                  *SPARSE_PATHS])
        if r.returncode != 0:
            shutil.rmtree(scratch, ignore_errors=True)
            raise RuntimeError(
                f"sparse-checkout failed:\n{r.stderr.strip()[:600]}")
    if not is_checkout(scratch):
        shutil.rmtree(scratch, ignore_errors=True)
        raise RuntimeError(
            f"fetched GasBrineBench {ref} but {SENTINEL} is missing -- "
            "the sparse paths or the layout changed")
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    scratch.rename(dest)
    return dest


def refresh_policy() -> str:
    """Return the GasBrineBench refresh policy set by ``ECPA_GBB_REFRESH``.

    Returns
    -------
    str
        ``"check"`` (the default when the variable is unset or empty),
        ``"always"`` or ``"never"``.  See the module description for what
        each one does in ``ensure``.

    Raises
    ------
    ValueError
        If ``ECPA_GBB_REFRESH`` holds any other value.
    """
    p = (os.environ.get("ECPA_GBB_REFRESH") or "check").strip().lower()
    if p not in ("check", "always", "never"):
        raise ValueError(
            f"ECPA_GBB_REFRESH={p!r}: expected 'check', 'always' or 'never'")
    return p


def ensure(refresh: str | None = None) -> Path:
    """Return a usable GasBrineBench checkout, fetching or refreshing it if needed.

    The checkout is chosen in this order:

    1. ``ECPA_GBB``, if set.  It is used as it is: never fetched and never
       checked against the remote.
    2. A directory named ``GasBrineBench`` in a parent directory of this
       package.
    3. The cache (``cache_dir``).
    4. If none exists, a fetch of ``ECPA_GBB_REF`` (default ``PINNED_REF``)
       into the cache with ``acquire``.

    Under the ``"check"`` policy the local commit is compared with the
    remote commit (``git ls-remote``, no download).  Only the cache is
    re-fetched when they differ; another checkout is left as it is and the
    difference is reported by ``describe``.  If a check or a fetch fails
    and a local copy exists, the local copy is used.

    Parameters
    ----------
    refresh : {"check", "always", "never"}, optional
        Refresh policy for this call.  Defaults to ``refresh_policy()``.
        Any value other than ``"check"`` or ``"always"`` acts as
        ``"never"``.

    Returns
    -------
    pathlib.Path
        Root directory of the checkout (it contains ``data/``).

    Raises
    ------
    RuntimeError
        If ``ECPA_GBB`` is set but does not hold a checkout, or if there is
        no local copy and the fetch fails.
    ValueError
        If ``refresh`` is None and ``ECPA_GBB_REFRESH`` is invalid.
    """
    global _RESOLVED, _HOW
    policy = (refresh or refresh_policy())
    ref = os.environ.get("ECPA_GBB_REF") or PINNED_REF

    pinned_path = os.environ.get("ECPA_GBB")
    local = find_local()

    if pinned_path:
        if local is None:
            raise RuntimeError(
                f"ECPA_GBB={pinned_path} does not hold {SENTINEL}. Unset it to "
                "search for a checkout, or point it at a real one.")
        _RESOLVED, _HOW = local, f"ECPA_GBB={local}"
        return local

    if policy == "always":
        try:
            p = acquire(cache_dir(), ref)
            _RESOLVED, _HOW = p, f"fetched {ref} (ECPA_GBB_REFRESH=always)"
            return p
        except Exception as exc:
            if local is None:
                raise
            _RESOLVED = local
            _HOW = f"{local} (forced fetch failed: {exc}; used local copy)"
            return local

    if local is None:
        p = acquire(cache_dir(), ref)
        _RESOLVED, _HOW = p, f"fetched {ref} into {p} (no local checkout)"
        return p

    if policy == "check":
        have, want = local_rev(local), remote_rev(ref)
        if want is None:
            _RESOLVED, _HOW = local, f"{local} (staleness check offline)"
        elif have == want:
            _RESOLVED, _HOW = local, f"{local} (up to date with {ref})"
        else:
            # A checkout the user maintains is theirs; say it is behind rather
            # than overwriting it. Only the cache we created is refreshed.
            if local == cache_dir():
                try:
                    p = acquire(cache_dir(), ref)
                    _RESOLVED, _HOW = p, f"refreshed cache to {ref}"
                    return p
                except Exception as exc:
                    _RESOLVED = local
                    _HOW = f"{local} (refresh failed: {exc})"
            else:
                _RESOLVED = local
                _HOW = (f"{local} (at {(have or '?')[:9]}, remote {ref} is "
                        f"{want[:9]} -- NOT refreshed, it is your working "
                        "checkout; git pull it or set ECPA_GBB_REFRESH=always)")
        return _RESOLVED

    _RESOLVED, _HOW = local, f"{local} (ECPA_GBB_REFRESH=never)"
    return local


def describe() -> str:
    """One line naming the database in use and how it was chosen.

    Resolves first if nothing has yet. A figure script that calls this to
    record its data provenance usually calls it BEFORE the first load, and
    the line then read "... [not resolved yet]" -- which says nothing about
    which database the figure was built from, the one thing it exists to say.
    """
    global _HOW
    if _RESOLVED is None:
        try:
            ensure()
        except Exception as exc:
            return f"GasBrineBench: not resolved ({exc})"
    root = _RESOLVED if _RESOLVED is not None else find_local()
    if root is None:
        return "GasBrineBench: not resolved"
    rev = local_rev(root)
    tag = _run(["git", "-C", str(root), "describe", "--tags", "--always"])
    label = tag.stdout.strip() if tag.returncode == 0 else (rev or "?")[:9]
    mark = ""
    if rev and rev != PINNED_COMMIT:
        mark = (f"  ** NOT the pinned revision {PINNED_COMMIT[:9]} the reported "
                "numbers were produced against **")
    return f"GasBrineBench {label} at {root} [{_HOW}]{mark}"


def load(family, *, exclude_tags="__default__", **kw):
    """Load GasBrineBench property families from the checkout given by ``ensure``.

    This calls ``gasbrinebench.load`` from that checkout, with
    ``where=<checkout>/data``.

    Parameters
    ----------
    family : str or sequence of str
        Property family name(s), or ``"all"``, as accepted by
        ``gasbrinebench.load``.
    exclude_tags : sequence of str or None
        Row tags to drop.  When not given, the default of
        ``gasbrinebench.load`` applies, which drops the 144 ``lle-regime``
        propane rows (liquid-liquid data).  Pass None to keep every row.
    **kw
        Further keyword arguments for ``gasbrinebench.load`` (for example
        ``gas="co2"``, or ``derive``).

    Returns
    -------
    pandas.DataFrame
        The selected rows, as returned by ``gasbrinebench.load``.

    Raises
    ------
    RuntimeError
        As for ``ensure``.
    """
    import sys
    root = ensure()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import gasbrinebench as _g
    if exclude_tags == "__default__":
        return _g.load(family, where=root / "data", **kw)
    return _g.load(family, where=root / "data", exclude_tags=exclude_tags, **kw)
