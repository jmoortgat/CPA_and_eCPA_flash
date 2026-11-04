"""Build the Quarto sources of the documentation website from README.md and docs/*.md.

The Markdown files on GitHub are the only sources. This script copies them into
site/_build/, with
  * README.md as the home page and every docs/*.md as one page (new files in
    docs/ appear on the site and in its navbar without further changes);
  * links rewritten: links between these pages point to the site pages, and
    links to any other file of the repository point to that file on GitHub;
  * at the top of each page, a link back to the same page on GitHub;
  * a generated _quarto.yml.

Usage (from the repository root):
    python3 site/build_site.py
    quarto render site/_build          # output in site/_build/_site/
"""
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "site" / "_build"
GITHUB = "https://github.com/jmoortgat/CPA_and_eCPA_flash"
SITE_URL = "https://jmoortgat.github.io/CPA_and_eCPA_flash/"
BRANCH = "main"

# source file (relative to the repository root) -> page file in the site
ORDER = ["papers.md", "models.md"]            # navbar order; other docs pages follow
PAGES = {"README.md": "index.md"}
for f in sorted((REPO / "docs").glob("*.md"),
                key=lambda f: (ORDER.index(f.name) if f.name in ORDER else len(ORDER), f.name)):
    PAGES[f"docs/{f.name}"] = f.name

LINK = re.compile(r"(\]\()([^)\s]+)(\))")
H1 = re.compile(r"^# (.+)$", re.M)


def resolve(target: str, source: str) -> str:
    """Rewrite one link target of the file `source`."""
    if re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
        return target
    path, _, anchor = target.partition("#")
    repo_path = (REPO / Path(source).parent / path).resolve().relative_to(REPO).as_posix()
    if repo_path in PAGES:
        url = PAGES[repo_path].replace(".md", ".html")
    elif (REPO / repo_path).is_dir():
        url = f"{GITHUB}/tree/{BRANCH}/{repo_path}".rstrip("/.")
    else:
        url = f"{GITHUB}/blob/{BRANCH}/{repo_path}"
    return url + (f"#{anchor}" if anchor else "")


def page(source: str) -> tuple[str, str]:
    text = (REPO / source).read_text()
    # the home page does not link to itself
    text = "\n".join(l for l in text.splitlines() if SITE_URL not in l) + "\n"
    m = H1.search(text)
    title = m.group(1).strip() if m else Path(source).stem
    if m:
        text = text[:m.start()] + text[m.end():].lstrip("\n")
    text = LINK.sub(lambda m: m.group(1) + resolve(m.group(2), source) + m.group(3), text)
    back = (f"[View this page on GitHub]({GITHUB}/blob/{BRANCH}/{source})"
            "{.github-link}\n\n")
    front = f'---\ntitle: "{title}"\n---\n\n'
    return title, front + back + text


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    nav = []
    for source, target in PAGES.items():
        title, body = page(source)
        (OUT / target).write_text(body)
        nav.append((target, "Home" if target == "index.md" else title))
    shutil.copy(REPO / "site" / "styles.css", OUT / "styles.css")
    items = "\n".join(f"      - href: {t}\n        text: \"{n}\"" for t, n in nav)
    (OUT / "_quarto.yml").write_text(f"""project:
  type: website
  output-dir: _site

website:
  title: "CPA and eCPA flash"
  site-url: {SITE_URL}
  repo-url: {GITHUB}
  navbar:
    left:
{items}
    right:
      - text: "Documentation on GitHub"
        icon: github
        href: {GITHUB}/tree/{BRANCH}/docs
  page-footer:
    center: "Generated from the Markdown files of the [repository]({GITHUB}); edit those, not this site."

format:
  html:
    theme:
      light: cosmo
      dark: darkly
    css: styles.css
    toc: true
""")
    print(f"wrote {len(PAGES)} pages to {OUT}")


if __name__ == "__main__":
    main()
