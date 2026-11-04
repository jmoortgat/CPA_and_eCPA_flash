"""Build the Quarto sources of the documentation website from the repository.

The Markdown files and notebooks on GitHub are the only sources. This script
copies them into site/_build/, with
  * README.md as the home page and every docs/*.md as one page;
  * every notebooks/*.ipynb as one page, shown with its saved outputs (not
    executed), with its code cells folded;
  * links rewritten: links between site pages point to the site, and links to
    any other file of the repository point to that file on GitHub;
  * at the top of each page, a link back to the same file on GitHub (and, for
    a notebook, a link to open it in Colab);
  * an API reference generated from the docstrings by quartodoc, with the
    sections listed in site/api.yml;
  * a generated _quarto.yml.
New files in docs/ or notebooks/ appear on the site and in its navbar without
further changes.

Usage (from the repository root; needs quartodoc 0.11 and PyYAML):
    python3 site/build_site.py
    (cd site/_build && PYTHONPATH=$PWD/../../code:$PWD/../../multisalt quartodoc build)
    python3 site/build_site.py --title-api
    quarto render site/_build          # output in site/_build/_site/
"""
import json
import os
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "site" / "_build"
GITHUB = "https://github.com/jmoortgat/CPA_and_eCPA_flash"
COLAB = "https://colab.research.google.com/github/jmoortgat/CPA_and_eCPA_flash/blob"
SITE_URL = "https://jmoortgat.github.io/CPA_and_eCPA_flash/"
BRANCH = "main"

# source file (relative to the repository root) -> page file in the site
DOC_ORDER = ["papers.md", "models.md"]                 # navbar order; other docs pages follow
NB_TITLES = {                                          # navbar entries; other notebooks follow
    "ecpa_flash_paper.ipynb": "Paper I: executable paper",
    "ecpa_flash_tutorial.ipynb": "Paper I: tutorial",
    "ms_ecpa_parametrization_paper.ipynb": "Paper II: executable paper",
    "ms_ecpa_parametrization_tutorial.ipynb": "Paper II: tutorial",
    "ms_ecpa_flash_paper.ipynb": "Paper III: executable paper",
    "ms_ecpa_flash_tutorial.ipynb": "Paper III: tutorial",
}
PAGES = {"README.md": "index.md"}
for f in sorted((REPO / "docs").glob("*.md"),
                key=lambda f: (DOC_ORDER.index(f.name) if f.name in DOC_ORDER else len(DOC_ORDER), f.name)):
    PAGES[f"docs/{f.name}"] = f.name
NOTEBOOKS = sorted((REPO / "notebooks").glob("*.ipynb"),
                   key=lambda f: (list(NB_TITLES).index(f.name) if f.name in NB_TITLES else len(NB_TITLES), f.name))
for f in NOTEBOOKS:
    PAGES[f"notebooks/{f.name}"] = f"notebooks/{f.name}"
ASSETS = ["notebooks/images"]                          # copied into the site as they are

LINK = re.compile(r"(\]\()([^)\s]+)(\))")
H1 = re.compile(r"^# (.+)$", re.M)


def site_name(target: str) -> str:
    return re.sub(r"\.(md|ipynb)$", ".html", target)


def resolve(target: str, source: str) -> str:
    """Rewrite one link target of the file `source` (both relative to the repository)."""
    if re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
        return target
    path, _, anchor = target.partition("#")
    repo_path = (REPO / Path(source).parent / path).resolve().relative_to(REPO).as_posix()
    here = Path(PAGES[source]).parent
    if repo_path in PAGES:
        url = os.path.relpath(site_name(PAGES[repo_path]), here)
    elif any(repo_path == a or repo_path.startswith(a + "/") for a in ASSETS):
        url = os.path.relpath(repo_path, here)
    elif (REPO / repo_path).is_dir():
        url = f"{GITHUB}/tree/{BRANCH}/{repo_path}".rstrip("/.")
    else:
        url = f"{GITHUB}/blob/{BRANCH}/{repo_path}"
    return url + (f"#{anchor}" if anchor else "")


def is_notebook(url: str) -> bool:
    return (url.split("#")[0].endswith(".ipynb") or "colab.research.google.com/github" in url
            or re.search(r"(^|/)notebooks/[^/#]+\.html", url) is not None
            or re.fullmatch(r"[^/:#]+_paper\.html|[^/:#]+_tutorial\.html", url.split("#")[0]) is not None)


NEW_TAB = '{target="_blank"}'


def relink(text: str, source: str) -> str:
    """Rewrite the links of a page; links to notebooks (site page, GitHub, Colab) open a new tab."""
    def one(m):
        url = resolve(m.group(2), source)
        tail = text[m.end():m.end() + 1]
        attr = NEW_TAB if is_notebook(url) and tail != "{" else ""
        return m.group(1) + url + m.group(3) + attr
    return LINK.sub(one, text)


def md_page(source: str) -> tuple[str, str]:
    text = (REPO / source).read_text()
    # the home page does not link to itself
    text = "\n".join(l for l in text.splitlines() if SITE_URL not in l) + "\n"
    m = H1.search(text)
    title = m.group(1).strip() if m else Path(source).stem
    if m:
        text = text[:m.start()] + text[m.end():].lstrip("\n")
    back = (f"[View this page on GitHub]({GITHUB}/blob/{BRANCH}/{source})"
            "{.github-link}\n\n")
    return title, f'---\ntitle: "{title}"\n---\n\n' + back + relink(text, source)


def notebook_page(source: str) -> dict:
    nb = json.loads((REPO / source).read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            cell["source"] = relink("".join(cell["source"]), source)
    # the first heading becomes the page title
    first = "".join(nb["cells"][0]["source"])
    m = H1.search(first) if nb["cells"][0]["cell_type"] == "markdown" else None
    title = m.group(1).strip() if m else Path(source).stem
    if m:
        nb["cells"][0]["source"] = first[:m.start()] + first[m.end():].lstrip("\n")
    links = (f"[View this notebook on GitHub]({GITHUB}/blob/{BRANCH}/{source}){{.github-link target=\"_blank\"}} "
             f"[Open in Colab]({COLAB}/{BRANCH}/{source}){{.github-link target=\"_blank\"}}\n\n"
             "This page shows the notebook with the outputs of its last run; open it in "
             "Colab or Jupyter to run the code.")
    # browser-tab title without LaTeX: $\mathrm{CO_{2}}$ -> CO2
    plain = re.sub(r"\$([^$]*)\$", lambda m: re.sub(r"\\[A-Za-z]+|[_^{}]", "", m.group(1)), title)
    front = {"cell_type": "raw", "metadata": {},
             "source": "---\ntitle: " + json.dumps(title) + "\npagetitle: " + json.dumps(plain) + "\n---"}
    intro = {"cell_type": "markdown", "metadata": {}, "source": links}
    nb["cells"] = [front, intro] + nb["cells"]
    for cell in nb["cells"]:
        cell.pop("id", None)
    nb["nbformat_minor"] = max(nb.get("nbformat_minor", 4), 4)
    return nb


def api_config() -> str:
    """The quartodoc block of _quarto.yml, from site/api.yml (written as JSON, which is YAML)."""
    import yaml
    sections = yaml.safe_load((REPO / "site" / "api.yml").read_text())
    cfg = {"package": None, "dir": "api", "title": "API reference",
           "sidebar": "api/_sidebar.yml", "parser": "numpy",
           "render_interlinks": False, "sections": sections}
    return "quartodoc: " + json.dumps(cfg, indent=2)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    docs, nbs = [], []
    for source, target in PAGES.items():
        dst = OUT / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        if source.endswith(".ipynb"):
            dst.write_text(json.dumps(notebook_page(source), indent=1))
            nbs.append((target, NB_TITLES.get(Path(source).name, Path(source).stem)))
        else:
            title, body = md_page(source)
            dst.write_text(body)
            docs.append((target, "Home" if target == "index.md" else title))
    for a in ASSETS:
        if (REPO / a).exists():
            shutil.copytree(REPO / a, OUT / a)
    shutil.copy(REPO / "site" / "styles.css", OUT / "styles.css")
    api = api_config()
    doc_items = "\n".join(f"      - href: {t}\n        text: \"{n}\"" for t, n in docs)
    nb_items = "\n".join(f"          - href: {t}\n            text: \"{n}\"\n            target: _blank" for t, n in nbs)
    (OUT / "_quarto.yml").write_text(f"""project:
  type: website
  output-dir: _site

execute:
  enabled: false                 # notebooks are shown with their saved outputs

website:
  title: "CPA and eCPA flash"
  site-url: {SITE_URL}
  repo-url: {GITHUB}
  navbar:
    left:
{doc_items}
      - text: "Notebooks"
        menu:
{nb_items}
      - href: api/index.qmd
        text: "API reference"
    right:
      - text: "Documentation on GitHub"
        icon: github
        href: {GITHUB}/tree/{BRANCH}/docs
  page-footer:
    center: "Generated from the Markdown files and notebooks of the [repository]({GITHUB}); edit those, not this site."

metadata-files:
  - api/_sidebar.yml

{api}

format:
  html:
    theme:
      light: cosmo
      dark: darkly
    css: styles.css
    toc: true
    code-fold: true
    code-summary: "Code"
""")
    print(f"wrote {len(docs)} pages and {len(nbs)} notebooks to {OUT}")


def title_api_pages():
    """Give each page that quartodoc wrote a browser-tab title from its heading."""
    for f in sorted((OUT / "api").glob("*.qmd")):
        text = f.read_text()
        if text.startswith("---"):
            continue
        m = re.match(r"# (.+?)\s*(\{[^}]*\})?\s*$", text.splitlines()[0])
        if m:
            f.write_text(f"---\npagetitle: {json.dumps(m.group(1))}\n---\n\n" + text)


if __name__ == "__main__":
    import sys
    if "--title-api" in sys.argv:          # run after `quartodoc build`
        title_api_pages()
    else:
        main()
