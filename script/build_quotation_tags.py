"""Per-span closer/opener tag table for Quotation spans, all 266 books."""
import re, csv, json, collections, statistics as st
from pathlib import Path

ROOT = Path.home()/"mnt"/"quotation_detection"
REPOS = ROOT/"Data"/"repos"
OUT   = ROOT/"Data"/"audit"/"quotation_tags"
OUT.mkdir(parents=True, exist_ok=True)

BLOCK = re.compile(r"^  (?P<i>[a-f0-9]+):\n    span:\n      start: (?P<s>-?\d+)\n      end: (?P<e>-?\d+)", re.M)

import sys
sys.path.insert(0, str(ROOT / "src"))
from layer_detection.quote_markers import (  # noqa: E402
    CLOSER_PROFILES, OPENER_PROFILES, CLOSER_WINDOW, OPENER_WINDOW,
    RECOMMENDED_CLOSER, RECOMMENDED_OPENER,
)
RE_DEFAULT = CLOSER_PROFILES[RECOMMENDED_CLOSER]
RE_OPENER  = OPENER_PROFILES[RECOMMENDED_OPENER]
AFTER_TOL, BEFORE_TOL, PREV = 40, 40, 60

split = {r["book"]: r["split"] for r in csv.DictReader(open(ROOT/"Data/audit/split.csv"))}

# ---- metadata (series / publisher / author / genre) -------------------------
import yaml
SUF = re.compile(r"[–—-]\s*(Book|དེབ།).*$")
SMERGE = {"དིལ་མགོ་མཁྱེན་བཙེ་གསུང་འབུམ།":"Dilgo Khyentse Series","ཞབས་དཀར་གསུང་འབུམ།":"Shabkar Series",
          "ཤེས་བྱ་ཀུན་ཁྱབ་མཛོད།":"Treasury of Knowledge","ཐོས་བསམ་འཆད་ཉན་གྱི་རིམ་པ།":"RDI Study Series",
          "རིག་པའི་རྡོ་རྗེ་གཙུག་ལག་སློབ་ཉེར་ཁང་།":"Rigpe Dorje Institute"}
PMERGE = {"Gautam Buddha ViharaDOWNLOAD EBOOK":"Gautam Buddha Vihara",
          "རིག་པའི་རྡོ་རྗེ་གཙུག་ལག་སློབ་ཉེར་ཁང་།":"Rigpe Dorje Institute",
          "བེན་ཆེན་སློབ་གཉེར་ཁང་།":"Benchen Publications",
          "Lekshey Ling Philosophy Series":"Lekshey Ling Publications"}
def meta_of(bd):
    m = bd/f"{bd.name}.opf"/"meta.yml"
    if not m.exists(): return {}
    d = yaml.safe_load(m.read_text(encoding="utf-8")) or {}
    w, s = d.get("web_metadata") or {}, d.get("source_metadata") or {}
    def g(k):
        v = w.get(k)
        return (v.get("en") or v.get("bo") or "").strip() if isinstance(v, dict) else ""
    coll = g("collection")
    ser = re.sub(r"\s+", " ", SUF.sub("", coll).strip(" –—-")) if coll else ""
    pub = g("pusblisher") or g("publisher")
    return dict(series=SMERGE.get(ser, ser), publisher=PMERGE.get(pub, pub),
                author=g("author") or ((s.get("authors") or [""])[0]),
                category=g("category"), sku=str(w.get("sku") or ""))

def clip(t, a, b):
    return t[max(a,0):max(b,0)].replace("\n", "⏎").replace("\r", "")

span_rows, book_rows = [], []
marker_counts = collections.Counter()

for bd in sorted(REPOS.iterdir()):
    if not bd.is_dir(): continue
    opf = bd/f"{bd.name}.opf"
    base, lay = opf/"base"/"v001.txt", opf/"layers"/"v001"/"Quotation.yml"
    if not base.exists(): continue
    md = meta_of(bd)
    text = base.read_text(encoding="utf-8", errors="replace")
    raw = []
    if lay.exists():
        raw = [(m["i"], int(m["s"]), int(m["e"])) for m in BLOCK.finditer(
            lay.read_text(encoding="utf-8", errors="replace"))]
    spans = [(i,s,e) for i,s,e in raw if e >= s]
    degen = len(raw) - len(spans)

    per = []
    for ann, s, e in sorted(spans, key=lambda x: x[1]):
        tail = clip(text, e-30, e+AFTER_TOL+10)
        head = clip(text, s-BEFORE_TOL, s+10)
        cw = clip(text, e+CLOSER_WINDOW[0], e+CLOSER_WINDOW[1])
        ow = clip(text, s+OPENER_WINDOW[0], s+OPENER_WINDOW[1])
        cm = {k: rx.search(cw) for k, rx in CLOSER_PROFILES.items()}
        om = {k: rx.search(ow) for k, rx in OPENER_PROFILES.items()}
        mD, mO = cm[RECOMMENDED_CLOSER], om[RECOMMENDED_OPENER]
        # distance from span end to the matched default closer, in chars
        dist = ""
        if mD:
            dist = (max(e-30,0) + mD.start()) - e
            marker_counts[mD.group(0)] += 1
        row = dict(
            book=bd.name, split=split.get(bd.name,"?"),
            series=md.get("series",""), publisher=md.get("publisher",""),
            author=md.get("author",""), category=md.get("category",""),
            ann_id=ann, start=s, end=e, length=e-s+1,
            **{f"closer_{k}": int(bool(v)) for k, v in cm.items()},
            **{f"opener_{k}": int(bool(v)) for k, v in om.items()},
            has_closer=int(bool(mD)),
            closer_marker=mD.group(0) if mD else "",
            closer_offset_from_end=dist,
            has_opener=int(bool(mO)), opener_marker=mO.group(0) if mO else "",
            marked=int(bool(mD or mO)),
            text_head=clip(text, s, s+PREV),
            text_tail=clip(text, e-PREV+1, e+1),
            after_span=clip(text, e+1, e+1+PREV),
            before_span=clip(text, s-PREV, s),
        )
        span_rows.append(row); per.append(row)

    n = len(per)
    if n:
        lens = [r["length"] for r in per]
        f = lambda k: sum(r[k] for r in per)/n
        mr = f("marked")
        band = "0-40%" if mr<.4 else "40-70%" if mr<.7 else "70-90%" if mr<.9 else "90-100%"
    else:
        lens, f, mr, band = [], lambda k: 0.0, "", "no-spans"
    qchars = sum(lens)
    book_rows.append(dict(
        book=bd.name, split=split.get(bd.name,"?"),
        series=md.get("series",""), publisher=md.get("publisher",""),
        author=md.get("author",""), category=md.get("category",""), sku=md.get("sku",""),
        doc_chars=len(text), n_spans=n, degenerate_spans=degen,
        quot_chars=qchars, density_pct=round(100*qchars/max(len(text),1), 3),
        median_span_len=(st.median(lens) if lens else ""),
        mean_span_len=(round(st.mean(lens),1) if lens else ""),
        **{f"rate_closer_{k}": (round(f(f"closer_{k}"),4) if n else "") for k in CLOSER_PROFILES},
        **{f"rate_opener_{k}": (round(f(f"opener_{k}"),4) if n else "") for k in OPENER_PROFILES},
        closer_rate=round(f("has_closer"),4) if n else "",
        opener_rate=round(f("has_opener"),4) if n else "",
        marked_rate=round(mr,4) if n else "",
        marked_band=band,
        reliable=int(n >= 10),
    ))

def write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return path

write(OUT/"quotation_spans_tags.csv", span_rows, list(span_rows[0]))
write(OUT/"quotation_book_tags.csv", book_rows, list(book_rows[0]))
write(OUT/"closer_marker_counts.csv",
      [dict(marker=m, count=c) for m,c in marker_counts.most_common()], ["marker","count"])

json.dump(dict(
    closer_profiles={k: v.pattern for k, v in CLOSER_PROFILES.items()},
    opener_profiles={k: v.pattern for k, v in OPENER_PROFILES.items()},
    recommended=dict(closer=RECOMMENDED_CLOSER, opener=RECOMMENDED_OPENER),
    windows=dict(closer=list(CLOSER_WINDOW), opener=list(OPENER_WINDOW)),
    totals=dict(books=len(book_rows), spans=len(span_rows),
                **{f"closer_{k}": sum(r[f"closer_{k}"] for r in span_rows) for k in CLOSER_PROFILES},
                **{f"opener_{k}": sum(r[f"opener_{k}"] for r in span_rows) for k in OPENER_PROFILES},
                marked=sum(r["marked"] for r in span_rows)),
), open(OUT/"tags_summary.json","w"), ensure_ascii=False, indent=2)
print("spans:", len(span_rows), " books:", len(book_rows))
print("wrote:", OUT)
