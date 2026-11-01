"""Build TECHNICAL.pdf from TECHNICAL.md: Markdown -> print-ready HTML -> headless Chrome/Edge PDF.

Run with the engine's core Python (it has the `markdown` package):
    engine\\envs\\core\\Scripts\\python.exe tools\\build_technical_pdf.py [--input TECHNICAL.md] [--output TECHNICAL.pdf]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parent.parent

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10pt; line-height: 1.45;
       color: #1d1f23; margin: 0; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center;
         border-left: 6px solid #2f6f4f; padding-left: 14mm; page-break-after: always; }
.cover h1 { font-size: 30pt; margin: 0 0 4mm; color: #16382a; border: 0; }
.cover .sub { font-size: 13pt; color: #4a5560; margin-bottom: 14mm; }
.cover .meta { font-size: 9.5pt; color: #6a737d; }
.tocpage { page-break-after: always; }
.tocpage h2 { margin-top: 0; }
.toc ul { list-style: none; padding-left: 0; margin: 0; }
.toc > ul > li { margin: 3mm 0 1mm; font-weight: 600; }
.toc ul ul { padding-left: 6mm; font-weight: 400; }
.toc ul ul li { margin: 0.6mm 0; }
.toc a { color: #1d1f23; text-decoration: none; }
h1 { font-size: 19pt; color: #16382a; border-bottom: 2px solid #2f6f4f; padding-bottom: 2mm;
     margin: 0 0 5mm; }
h1.part { page-break-before: always; }
h2 { font-size: 14pt; color: #16382a; margin: 7mm 0 2.5mm; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #2f6f4f; margin: 5mm 0 2mm; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 4mm 0 1.5mm; page-break-after: avoid; }
p { margin: 0 0 2.5mm; }
ul, ol { margin: 0 0 2.5mm; padding-left: 6mm; }
li { margin: 0.8mm 0; }
a { color: #2f6f4f; }
code { font-family: Consolas, "Cascadia Mono", monospace; font-size: 8.8pt; background: #f1f3f2;
       padding: 0.2mm 1mm; border-radius: 2px; }
pre { background: #f4f6f5; border: 1px solid #dde3df; border-left: 3px solid #2f6f4f;
      padding: 3mm 3mm; border-radius: 3px; overflow: hidden; white-space: pre;
      page-break-inside: avoid; margin: 2mm 0 3.5mm; }
pre code { background: none; padding: 0; font-size: 7.6pt; line-height: 1.4; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 4mm; font-size: 9pt;
        page-break-inside: avoid; }
th { background: #e6efe9; color: #16382a; text-align: left; }
th, td { border: 1px solid #cfd8d2; padding: 1.4mm 2.2mm; vertical-align: top; }
tr:nth-child(even) td { background: #fafbfa; }
hr { display: none; }
strong { color: #111; }
"""


def build_html(src: Path) -> str:
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = lines[0].lstrip("# ").strip()   # "PlateCraft — technical description ..."
    name = title.split(" — ")[0]

    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"],
                           extension_configs={"toc": {"toc_depth": "1-2"}})
    body = md.convert("\n".join(lines[1:]))
    # "# Part A — ..." headings start a new page.
    body = re.sub(r"<h1 ", '<h1 class="part" ', body)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head>
<body>
<section class="cover">
  <h1>{name}</h1>
  <div class="sub">Technical description and production guide<br>
  Video to PBR material pipeline for Unreal Engine 5.8</div>
  <div class="meta">Generated from {src.name}</div>
</section>
<section class="tocpage"><h2>Contents</h2>{md.toc}</section>
{body}
</body></html>"""


def find_browser() -> str | None:
    candidates = []
    for env in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if base:
            candidates += [Path(base) / "Google/Chrome/Application/chrome.exe",
                           Path(base) / "Microsoft/Edge/Application/msedge.exe"]
    for c in candidates:
        if c.is_file():
            return str(c)
    for exe in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
        found = shutil.which(exe)
        if found:
            return found
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=REPO / "TECHNICAL.md")
    ap.add_argument("--output", type=Path, default=REPO / "TECHNICAL.pdf")
    ap.add_argument("--browser", help="Chrome/Edge/Chromium executable (default: search)")
    args = ap.parse_args()

    browser = args.browser or find_browser()
    if not browser:
        print("error: no Chrome, Edge or Chromium found; pass --browser", file=sys.stderr)
        return 2

    out = args.output.resolve()
    # Chrome's crash handler can hold a profile file a moment after exit: don't fail on cleanup.
    with tempfile.TemporaryDirectory(prefix="platecraft-pdf-", ignore_cleanup_errors=True) as tmp:
        html = Path(tmp) / "technical.html"
        html.write_text(build_html(args.input), encoding="utf-8")
        out.unlink(missing_ok=True)
        # A throwaway profile, so a running browser isn't reused and nothing lands in the user's profile.
        subprocess.run([browser, "--headless", "--disable-gpu", f"--user-data-dir={Path(tmp) / 'profile'}",
                        "--no-pdf-header-footer", f"--print-to-pdf={out}", html.as_uri()],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not out.is_file():
        print(f"error: {Path(browser).name} wrote no PDF", file=sys.stderr)
        return 1
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
