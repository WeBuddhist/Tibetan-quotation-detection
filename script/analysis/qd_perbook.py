import re, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPOS = ROOT/"Data"/"repos"
LAYERS = ("Quotation","Sabche","Tsawa","Yigchung","Chapter","Author","BookTitle")
BLOCK = re.compile(r"^  (?P<i>[a-f0-9]+):\n    span:\n      start: (?P<s>-?\d+)\n      end: (?P<e>-?\d+)", re.M)
LIST  = re.compile(r"^- span:\n    start: (?P<s>-?\d+)\n    end: (?P<e>-?\d+)", re.M)

split = {r["book"]: r["split"] for r in csv.DictReader(open(ROOT/"Data/audit/split.csv"))}

def spans(p):
    t = p.read_text(encoding="utf-8", errors="replace")
    out = [(int(m["s"]), int(m["e"])) for m in BLOCK.finditer(t)]
    if not out:
        out = [(int(m["s"]), int(m["e"])) for m in LIST.finditer(t)]
    return [(s,e) for s,e in out if e >= s]   # drop inverted/empty (the cleaning rule)

rows=[]
for bd in sorted(REPOS.iterdir()):
    if not bd.is_dir(): continue
    opf = bd/f"{bd.name}.opf"
    base = opf/"base"/"v001.txt"
    lay  = opf/"layers"/"v001"
    if not base.exists() or not lay.exists(): continue
    doc = base.read_text(encoding="utf-8", errors="replace")
    n = len(doc)
    row = {"book": bd.name, "split": split.get(bd.name,"?"), "doc_len": n}
    for L in LAYERS:
        f = lay/f"{L}.yml"
        sp = spans(f) if f.exists() else []
        row[L+"_n"] = len(sp)
        row[L+"_chars"] = sum(e-s+1 for s,e in sp)
    rows.append(row)

json.dump(rows, open(Path.home()/"qd/perbook.json","w"))
print("books:", len(rows))
