#!/usr/bin/env python3
"""Render one book's quotation annotations as a standalone HTML page you can read.

Built to answer "what is wrong with this book" at a glance. Every quotation span
is colour-coded by its opener class and flagged when it looks broken, so a
mis-annotated book (P000115 marks source references, not quotations) is obvious
within seconds of opening the page.

    python script/quotation_inspector.py P000115
    python script/quotation_inspector.py I25463E37 --out /tmp/x.html
    python script/quotation_inspector.py --index            # all books, ranked by problems
    python script/quotation_inspector.py --index --min-spans 20 --open

Handles both annotation conventions: `Quotation.yml` in the P0* batch and
`Citation.yml` in the I* batch (shown to be the same annotation -- see
docs/handoff-quotation-split-cursor.md §9.2).

No dependencies beyond the standard library. Output is one self-contained file.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOS = REPO_ROOT / "Data" / "repos"
DEFAULT_OUT = REPO_ROOT / "Data" / "audit" / "quotation_views"

BLOCK = re.compile(
    r"^  (?P<ann>[a-f0-9]+):\n    span:\n      start: (?P<s>-?\d+)\n      end: (?P<e>-?\d+)",
    re.M,
)
DELIM = "་༌ \t\n\r།༎༏༑༈༔༴༵༶྅"
PUN = set(DELIM)
SHAD = set("།༎༏༑༈༔")
SPLIT = re.compile("[" + re.escape(DELIM) + "]+")

SPEECH = {"གསུངས", "སྩལ", "སྨྲས", "བཤད", "བརྗོད", "ཞུས", "གསོལ", "ཟེར"}
NOMZ = {"པ", "པར", "བ", "བར", "པོ", "པའི", "བའི"}
ERG_EXPLICIT = {"གིས", "ཀྱིས", "གྱིས", "ཡིས"}
ERG_NAME = re.compile(r"(?:པས|པོས|བས|ཆེས|ལྷས|ཧས|བོས|མས|རས|ཏིས|བུས|རྗེས|མཚོས|གྲགས|གེས|ཤས)$")
TERMINATIVE = {"དུ", "ཏུ", "རུ", "སུ"}
MARKER_SYLLABLES = {"ལས", "ནས"} | TERMINATIVE | ERG_EXPLICIT

CLASS_COLOUR = {
    "TEXT":    ("#1d4ed8", "#dbeafe"),
    "PERSON":  ("#047857", "#d1fae5"),
    "FORMULA": ("#7c3aed", "#ede9fe"),
    "CHAINED": ("#b45309", "#fef3c7"),
    "MISC":    ("#9f1239", "#ffe4e6"),
}
CLASS_LABEL = {
    "TEXT": "TEXT · from a book (ལས/ནས)",
    "PERSON": "PERSON · from a person (-ས/ཞལ་ནས)",
    "FORMULA": "FORMULA · as it is said (སྐད་དུ)",
    "CHAINED": "CHAINED · …and (ཞེས་དང)",
    "MISC": "MISC · no opener found",
}


def syllables(text: str) -> list[str]:
    return [p for p in SPLIT.split(text) if p]


def preceding(text: str, start: int, count: int = 2) -> tuple[list[str], bool]:
    """The `count` syllables before `start`, and whether a shad was crossed."""
    i = start - 1
    crossed_shad = False
    while i >= 0 and text[i] in PUN:
        if text[i] in SHAD:
            crossed_shad = True
        i -= 1
    out: list[str] = []
    for _ in range(count):
        if i < 0:
            break
        j = i
        while j >= 0 and text[j] not in PUN:
            j -= 1
        out.append(text[j + 1 : i + 1])
        i = j - 1
        while i >= 0 and text[i] in PUN:
            i -= 1
    return (out + [""] * count)[:count], crossed_shad


def classify(prev: str, before_prev: str) -> str:
    if prev == "རེ" and before_prev == "ན":
        return "PERSON"
    if prev == "དང" and before_prev in ("ཞེས", "ཅེས"):
        return "CHAINED"
    if prev in NOMZ and before_prev in SPEECH:
        return "PERSON"
    if prev in ("ཀྱང", "ཡང") and before_prev:
        prev = before_prev
    if prev in ("ལས", "ནས"):
        return "PERSON" if before_prev in ("ཞལ", "གསུང") else "TEXT"
    if prev in SPEECH or prev in ERG_EXPLICIT or ERG_NAME.search(prev):
        return "PERSON"
    if prev in TERMINATIVE:
        return "FORMULA"
    if prev in ("ཞེས", "ཅེས"):
        return "CHAINED"
    return "MISC"


def layer_file(opf: Path) -> tuple[Path, str] | tuple[None, None]:
    for name in ("Quotation", "Citation"):
        path = opf / "layers" / "v001" / f"{name}.yml"
        if path.is_file():
            return path, name
    return None, None


def analyse(book: str, repos: Path) -> dict | None:
    opf = repos / book / f"{book}.opf"
    base = opf / "base" / "v001.txt"
    path, layer = layer_file(opf)
    if not (base.is_file() and path):
        return None
    text = base.read_text(encoding="utf-8", errors="replace")
    spans = []
    for m in BLOCK.finditer(path.read_text(encoding="utf-8", errors="replace")):
        s, e = int(m["s"]), int(m["e"])
        if e < s:
            spans.append({"ann": m["ann"], "start": s, "end": e, "degenerate": True})
            continue
        (prev, before_prev), new_sentence = preceding(text, s)
        body = text[s : e + 1]
        tail = syllables(body[-40:])
        head = syllables(body[:60])[:4]
        flags = []
        if s > 0 and text[s - 1] not in PUN:
            flags.append("start cuts a syllable")
        if e + 1 < len(text) and text[e + 1] not in PUN:
            flags.append("end cuts a syllable")
        if len(body) <= 30 and tail and tail[-1] in MARKER_SYLLABLES:
            flags.append("looks like the introducer, not the quote")
        if any(x in MARKER_SYLLABLES for x in head):
            flags.append("marker sits inside the span")
        if not new_sentence:
            flags.append("starts mid-sentence")
        after = syllables(text[e + 1 : e + 40])
        spans.append({
            "ann": m["ann"], "start": s, "end": e, "length": e - s + 1,
            "cls": classify(prev, before_prev), "prev": prev, "before_prev": before_prev,
            "new_sentence": new_sentence, "flags": flags,
            "closed_by": after[0] if after and after[0] in ("ཞེས", "ཅེས", "གསུངས") else "",
            "degenerate": False,
        })
    spans.sort(key=lambda x: x["start"])
    return {"book": book, "layer": layer, "text": text, "spans": spans}


def summarise(data: dict) -> dict:
    spans = [s for s in data["spans"] if not s["degenerate"]]
    n = len(spans)
    if not n:
        return {"n": 0}
    counts: dict[str, int] = {}
    for s in spans:
        counts[s["cls"]] = counts.get(s["cls"], 0) + 1
    flagged = sum(1 for s in spans if s["flags"])
    return {
        "n": n,
        "degenerate": sum(1 for s in data["spans"] if s["degenerate"]),
        "median_len": statistics.median(s["length"] for s in spans),
        "counts": counts,
        "flagged": flagged,
        "flagged_pct": 100 * flagged / n,
        "introducer_only": sum(1 for s in spans if "looks like the introducer, not the quote" in s["flags"]),
        "misaligned": sum(1 for s in spans if any("cuts a syllable" in f for f in s["flags"])),
        "mid_sentence": sum(1 for s in spans if not s["new_sentence"]),
        "closed": sum(1 for s in spans if s["closed_by"]),
        "doc_chars": len(data["text"]),
    }


CSS = """
:root{--bg:#fbfaf8;--fg:#1c1917;--mut:#78716c;--line:#e7e5e4;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--line:#44403c;--card:#292524}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 420px;height:100vh}
@media(max-width:1100px){.wrap{grid-template-columns:1fr;height:auto}}
header{padding:14px 20px;border-bottom:1px solid var(--line);background:var(--card);
 position:sticky;top:0;z-index:5}
h1{margin:0 0 4px;font-size:17px;font-weight:600}
.sub{color:var(--mut);font-size:12.5px}
.stats{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.pill{padding:3px 9px;border-radius:99px;font-size:12px;border:1px solid var(--line);
 background:var(--bg);cursor:pointer;user-select:none}
.pill.on{outline:2px solid currentColor;outline-offset:-2px;font-weight:600}
.pill.warn{background:#fee2e2;color:#991b1b;border-color:#fca5a5}
@media (prefers-color-scheme:dark){.pill.warn{background:#450a0a;color:#fca5a5;border-color:#7f1d1d}}
main{overflow:auto;padding:20px 26px}
.tib{font-family:"Noto Serif Tibetan","Microsoft Himalaya","Kailasa","Jomolhari",serif;
 font-size:19px;line-height:2.5;white-space:pre-wrap;word-break:break-word}
mark{padding:2px 0;border-radius:3px;cursor:pointer;border-bottom:2px solid}
mark.flagged{outline:2px dashed #dc2626;outline-offset:1px}
aside{border-left:1px solid var(--line);overflow:auto;background:var(--card)}
.row{padding:10px 14px;border-bottom:1px solid var(--line);cursor:pointer;font-size:12.5px}
.row:hover{background:var(--bg)}
.row .cls{font-weight:600;font-size:11px;letter-spacing:.03em}
.row .meta{color:var(--mut);margin-top:2px}
.row .snip{font-family:"Noto Serif Tibetan","Microsoft Himalaya",serif;font-size:15px;
 margin-top:4px;line-height:1.9;color:var(--fg)}
.flag{display:inline-block;background:#fee2e2;color:#991b1b;border-radius:3px;
 padding:1px 5px;margin:2px 3px 0 0;font-size:10.5px}
@media (prefers-color-scheme:dark){.flag{background:#450a0a;color:#fca5a5}}
.hidden{display:none!important}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}
a{color:inherit}
.bad{color:#dc2626;font-weight:600}
"""

JS = """
const pills=[...document.querySelectorAll('.pill[data-filter]')];
let active=null;
function apply(){
  document.querySelectorAll('.row').forEach(r=>{
    const ok = !active || (active==='FLAGGED' ? r.dataset.flagged==='1' : r.dataset.cls===active);
    r.classList.toggle('hidden',!ok);
  });
  document.querySelectorAll('mark').forEach(m=>{
    const ok = !active || (active==='FLAGGED' ? m.dataset.flagged==='1' : m.dataset.cls===active);
    m.style.opacity = ok?'1':'0.18';
  });
}
pills.forEach(p=>p.onclick=()=>{
  const f=p.dataset.filter;
  active = (active===f)?null:f;
  pills.forEach(q=>q.classList.toggle('on',q.dataset.filter===active));
  apply();
});
function jump(id){
  const m=document.getElementById('s'+id);
  if(!m) return;
  m.scrollIntoView({behavior:'smooth',block:'center'});
  m.style.transition='none'; m.style.boxShadow='0 0 0 3px #dc2626';
  setTimeout(()=>{m.style.transition='box-shadow .8s';m.style.boxShadow='none';},700);
}
document.querySelectorAll('.row').forEach(r=>r.onclick=()=>jump(r.dataset.id));
document.querySelectorAll('mark').forEach(m=>m.onclick=()=>{
  const r=document.querySelector('.row[data-id="'+m.dataset.id+'"]');
  if(r){r.scrollIntoView({behavior:'smooth',block:'center'});
        r.style.background='#fde68a';setTimeout(()=>r.style.background='',900);}
});
"""


def render_book(data: dict, summary: dict) -> str:
    text, spans = data["text"], data["spans"]
    live = [s for s in spans if not s["degenerate"]]
    pieces, cursor = [], 0
    for i, sp in enumerate(live):
        if sp["start"] > cursor:
            pieces.append(html.escape(text[cursor : sp["start"]]))
        fg, bg = CLASS_COLOUR[sp["cls"]]
        flagged = "1" if sp["flags"] else "0"
        cls = " flagged" if sp["flags"] else ""
        pieces.append(
            f'<mark id="s{i}" class="{cls}" data-id="{i}" data-cls="{sp["cls"]}" '
            f'data-flagged="{flagged}" style="background:{bg};border-color:{fg}" '
            f'title="{sp["cls"]} · {sp["length"]} chars · prev={html.escape(sp["prev"])}">'
            f'{html.escape(text[sp["start"] : sp["end"] + 1])}</mark>'
        )
        cursor = sp["end"] + 1
    pieces.append(html.escape(text[cursor:]))

    rows = []
    for i, sp in enumerate(live):
        fg, _ = CLASS_COLOUR[sp["cls"]]
        flags = "".join(f'<span class="flag">{html.escape(f)}</span>' for f in sp["flags"])
        snippet = text[sp["start"] : sp["start"] + 70].replace("\n", " ")
        closed = f' · closed by {sp["closed_by"]}' if sp["closed_by"] else " · not closed"
        rows.append(
            f'<div class="row" data-id="{i}" data-cls="{sp["cls"]}" '
            f'data-flagged="{"1" if sp["flags"] else "0"}">'
            f'<div class="cls" style="color:{fg}">{sp["cls"]}</div>'
            f'<div class="meta">{sp["start"]}–{sp["end"]} · {sp["length"]} chars · '
            f'prev <b>{html.escape(sp["prev"]) or "—"}</b>{closed} · '
            f'{"sentence-initial" if sp["new_sentence"] else "mid-sentence"}</div>'
            f'<div class="snip">{html.escape(snippet)}…</div>{flags}'
            f'<div class="meta" style="font-size:11px;opacity:.7">{sp["ann"]}</div></div>'
        )

    counts = summary["counts"]
    pills = [
        f'<span class="pill" data-filter="{c}" style="color:{CLASS_COLOUR[c][0]}">'
        f'{c} {counts.get(c,0)}</span>'
        for c in ("TEXT", "PERSON", "FORMULA", "CHAINED", "MISC")
    ]
    pills.append(
        f'<span class="pill warn" data-filter="FLAGGED">⚠ flagged {summary["flagged"]} '
        f'({summary["flagged_pct"]:.0f}%)</span>'
    )
    warn = ""
    if summary["introducer_only"] > summary["n"] * 0.5:
        warn = ('<div class="pill warn" style="margin-top:8px;display:block">'
                'This book looks like it annotates <b>source references</b>, not quotations — '
                'most spans are just the citation introducer. Compare P000115.</div>')

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{data['book']} · quotation inspector</title><style>{CSS}</style></head><body>
<div class="wrap"><div style="display:flex;flex-direction:column;min-height:0">
<header><h1>{data['book']} <span style="font-weight:400;color:var(--mut)">·
 {data['layer']}.yml</span></h1>
<div class="sub">{summary['n']} spans · median {summary['median_len']:.0f} chars ·
 {summary['doc_chars']:,} chars of text ·
 {summary['misaligned']} with a cut syllable ·
 {summary['mid_sentence']} mid-sentence ·
 {summary['closed']} closed by ཞེས/ཅེས/གསུངས</div>
<div class="stats">{''.join(pills)}</div>{warn}</header>
<main><div class="tib">{''.join(pieces)}</div></main></div>
<aside>{''.join(rows)}</aside></div><script>{JS}</script></body></html>"""


def render_index(rows: list[dict]) -> str:
    body = []
    for r in rows:
        pct = r["flagged_pct"]
        cls = ' class="bad"' if pct > 40 else ""
        c = r["counts"]
        body.append(
            f'<tr><td><a href="{r["book"]}.html">{r["book"]}</a></td>'
            f'<td>{r["layer"]}</td>'
            f'<td class="num">{r["n"]}</td><td class="num">{r["median_len"]:.0f}</td>'
            f'<td class="num"{cls}>{pct:.0f}%</td>'
            f'<td class="num">{r["introducer_only"]}</td>'
            f'<td class="num">{100*r["misaligned"]/r["n"]:.0f}%</td>'
            f'<td class="num">{100*r["mid_sentence"]/r["n"]:.0f}%</td>'
            f'<td class="num">{100*c.get("TEXT",0)/r["n"]:.0f}%</td>'
            f'<td class="num">{100*c.get("PERSON",0)/r["n"]:.0f}%</td>'
            f'<td class="num">{100*c.get("MISC",0)/r["n"]:.0f}%</td></tr>'
        )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>quotation inspector · index</title><style>{CSS}
body{{padding:26px 32px;max-width:1150px;margin:0 auto}}</style></head><body>
<h1>Quotation annotation · {len(rows)} books</h1>
<p class="sub">Sorted by share of flagged spans. Click a book to inspect it.
A high <b>intro-only</b> count means the annotator marked the citation reference
rather than the quoted passage.</p>
<table><thead><tr><th>book</th><th>layer</th><th>spans</th><th>med len</th>
<th>flagged</th><th>intro-only</th><th>cut syllable</th><th>mid-sent</th>
<th>TEXT</th><th>PERSON</th><th>MISC</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book", nargs="?", help="book id, e.g. P000115 or I25463E37")
    ap.add_argument("--repos", type=Path, default=DEFAULT_REPOS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--index", action="store_true", help="render every book plus an index page")
    ap.add_argument("--min-spans", type=int, default=1)
    ap.add_argument("--open", action="store_true", help="open the result in a browser")
    args = ap.parse_args()

    out_dir = args.out if (args.out and args.out.is_dir()) else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.index:
        rows = []
        books = sorted(d.name for d in args.repos.iterdir() if d.is_dir())
        for n, book in enumerate(books, 1):
            data = analyse(book, args.repos)
            if not data:
                continue
            s = summarise(data)
            if s["n"] < args.min_spans:
                continue
            (out_dir / f"{book}.html").write_text(render_book(data, s), encoding="utf-8")
            rows.append({"book": book, "layer": data["layer"], **s})
            if n % 50 == 0:
                print(f"  … {n}/{len(books)}")
        rows.sort(key=lambda r: -r["flagged_pct"])
        index = out_dir / "index.html"
        index.write_text(render_index(rows), encoding="utf-8")
        print(f"wrote {len(rows)} book pages + {index}")
        if args.open:
            webbrowser.open(index.as_uri())
        return 0

    if not args.book:
        ap.error("give a book id, or use --index")
    data = analyse(args.book, args.repos)
    if not data:
        print(f"no quotation/citation layer for {args.book}")
        return 1
    s = summarise(data)
    path = args.out if (args.out and not args.out.is_dir()) else out_dir / f"{args.book}.html"
    path.write_text(render_book(data, s), encoding="utf-8")
    print(f"{args.book}: {s['n']} spans, {s['flagged']} flagged ({s['flagged_pct']:.0f}%), "
          f"median {s['median_len']:.0f} chars")
    print(f"wrote {path}")
    if args.open:
        webbrowser.open(path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
