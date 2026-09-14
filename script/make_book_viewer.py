#!/usr/bin/env python3
"""Render one book's annotations as a standalone HTML file you can open and read.

The point is to make the corpus inspectable by someone who is not going to run
Python -- a Tibetanist, an annotation lead -- so the output is a single file with
no server, no dependencies, and no build step.

    python script/make_book_viewer.py P000218
    python script/make_book_viewer.py P000078 --layers Quotation --gaps
    python script/make_book_viewer.py P000051 --out /tmp/p51.html

Opens with every annotated layer colour-coded over the raw text. `--gaps` marks
long unannotated stretches, which is how you check whether a book was abandoned
partway through rather than genuinely quotation-free.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAYERS = ["Quotation", "Sabche", "Tsawa", "Yigchung", "Chapter", "Author", "BookTitle"]
COLOURS = {
    "Quotation": "#ffd54f", "Sabche": "#81d4fa", "Tsawa": "#a5d6a7",
    "Yigchung": "#ce93d8", "Chapter": "#ffab91", "Author": "#b0bec5",
    "BookTitle": "#f48fb1",
}


def load_book(book: str, repos: Path):
    opf = repos / book / f"{book}.opf"
    text = (opf / "base" / "v001.txt").read_text(encoding="utf-8")
    spans = []
    for layer in LAYERS:
        path = opf / "layers" / "v001" / f"{layer}.yml"
        if not path.is_file():
            continue
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except ImportError:
            raise SystemExit("pip install pyyaml")
        for ann_id, body in (data.get("annotations") or {}).items():
            sp = (body or {}).get("span") or {}
            if "start" in sp and "end" in sp:
                spans.append(
                    {"layer": layer, "start": int(sp["start"]),
                     "end": int(sp["end"]), "id": str(ann_id)[:8]}
                )
    return text, spans


def segment(text_len: int, spans: list[dict]):
    """Split the document into runs that share the same set of active layers.

    Handles overlap without assuming there isn't any: a character can belong to
    several layers at once, and the run is keyed on the whole set.
    """
    edges = sorted({0, text_len} | {s["start"] for s in spans}
                   | {s["end"] + 1 for s in spans})
    edges = [e for e in edges if 0 <= e <= text_len]
    runs = []
    for a, b in zip(edges, edges[1:]):
        active = [s for s in spans if s["start"] <= a and s["end"] >= b - 1]
        runs.append((a, b, active))
    return runs


def build_html(book: str, text: str, spans: list[dict], gap_min: int) -> str:
    present = sorted({s["layer"] for s in spans}, key=LAYERS.index)
    runs = segment(len(text), spans)

    # long stretches with no annotation at all
    gaps = []
    if gap_min:
        cursor = 0
        for s in sorted(spans, key=lambda s: s["start"]):
            if s["start"] - cursor >= gap_min:
                gaps.append((cursor, s["start"]))
            cursor = max(cursor, s["end"] + 1)
        if len(text) - cursor >= gap_min:
            gaps.append((cursor, len(text)))

    body = []
    for a, b, active in runs:
        chunk = html.escape(text[a:b]).replace("\n", "<br>")
        in_gap = any(g0 <= a < g1 for g0, g1 in gaps)
        if not active:
            body.append(f'<span class="{"gap" if in_gap else "plain"}">{chunk}</span>')
            continue
        layer = active[0]["layer"]
        title = " + ".join(f'{s["layer"]} [{s["start"]}–{s["end"]}] {s["id"]}' for s in active)
        cls = "ann multi" if len(active) > 1 else "ann"
        body.append(
            f'<span class="{cls}" data-layer="{layer}" title="{html.escape(title)}" '
            f'style="background:{COLOURS.get(layer, "#eee")}">{chunk}</span>'
        )

    by_layer = {}
    for s in spans:
        by_layer.setdefault(s["layer"], []).append(s)
    rows = []
    for layer in present:
        items = by_layer[layer]
        lens = [s["end"] - s["start"] + 1 for s in items]
        degenerate = sum(1 for n in lens if n <= 2)
        covered = sum(lens)
        warn = ' <b style="color:#c62828">← degenerate</b>' if degenerate else ""
        rows.append(
            f"<tr><td><span class='dot' style='background:{COLOURS.get(layer)}'></span>{layer}</td>"
            f"<td>{len(items)}</td><td>{min(lens)}</td><td>{sum(lens)//len(lens)}</td>"
            f"<td>{max(lens)}</td><td>{100*covered/len(text):.2f}%</td>"
            f"<td>{degenerate}{warn}</td></tr>"
        )

    gap_rows = "".join(
        f"<tr><td>{g0:,}</td><td>{g1:,}</td><td>{g1-g0:,}</td>"
        f"<td>{100*(g1-g0)/len(text):.1f}%</td></tr>" for g0, g1 in gaps
    ) or "<tr><td colspan=4>none</td></tr>"

    toggles = "".join(
        f'<label><input type="checkbox" checked data-layer="{l}" onchange="tog(this)"> '
        f'<span class="dot" style="background:{COLOURS.get(l)}"></span>{l}</label> '
        for l in present
    )

    return f"""<!doctype html><meta charset="utf-8"><title>{book}</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:0;background:#fafafa}}
 header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;
   padding:14px 22px;box-shadow:0 1px 4px rgba(0,0,0,.06);z-index:9}}
 h1{{margin:0 0 6px;font-size:17px}} .meta{{color:#666;font-size:13px}}
 table{{border-collapse:collapse;font-size:13px;margin:10px 0}}
 th,td{{padding:3px 12px;text-align:left;border-bottom:1px solid #eee}}
 th{{color:#666;font-weight:600}}
 .dot{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}}
 label{{font-size:13px;margin-right:14px;cursor:pointer;white-space:nowrap}}
 #text{{padding:26px 34px;line-height:2.4;font-size:19px;
   font-family:"Noto Serif Tibetan","Jomolhari","Microsoft Himalaya","Kailasa",serif;
   word-break:break-word}}
 .ann{{border-radius:3px;padding:1px 0;cursor:help}}
 .multi{{outline:2px dashed #d32f2f;outline-offset:1px}}
 .gap{{background:repeating-linear-gradient(45deg,#fff,#fff 8px,#ffebee 8px,#ffebee 16px)}}
 .off{{background:transparent!important;outline:none!important}}
 details{{margin-top:8px;font-size:13px}}
</style>
<header>
 <h1>{book} <span class="meta">— {len(text):,} characters · {len(spans)} annotations</span></h1>
 <table><tr><th>layer</th><th>spans</th><th>min</th><th>mean</th><th>max</th>
   <th>coverage</th><th>len ≤ 2 chars</th></tr>{''.join(rows)}</table>
 <div>{toggles}</div>
 <details><summary>Unannotated stretches ≥ {gap_min:,} chars ({len(gaps)})</summary>
  <table><tr><th>from</th><th>to</th><th>length</th><th>of book</th></tr>{gap_rows}</table>
 </details>
</header>
<div id="text">{''.join(body)}</div>
<script>
function tog(cb){{
  document.querySelectorAll('[data-layer="'+cb.dataset.layer+'"].ann')
    .forEach(function(e){{ e.classList.toggle('off', !cb.checked); }});
}}
</script>"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("book")
    p.add_argument("--repos", type=Path, default=REPO_ROOT / "Data" / "repos")
    p.add_argument("--out", type=Path)
    p.add_argument("--layers", nargs="*", help="only render these layers")
    p.add_argument("--gap-min", type=int, default=50_000,
                   help="flag unannotated stretches at least this long (0 = off)")
    a = p.parse_args()

    text, spans = load_book(a.book, a.repos)
    if a.layers:
        spans = [s for s in spans if s["layer"] in a.layers]
    out = a.out or REPO_ROOT / f"viewer_{a.book}.html"
    out.write_text(build_html(a.book, text, spans, a.gap_min), encoding="utf-8")
    print(f"{out}   ({len(text):,} chars, {len(spans)} annotations)")
    print("open it in a browser — no server needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
