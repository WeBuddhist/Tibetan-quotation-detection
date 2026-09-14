"""Mine candidate opener/closer markers from TRAIN books only, by enrichment."""
import re, csv, json, collections, random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; REPOS=ROOT/"Data"/"repos"
BLOCK=re.compile(r"^  (?P<i>[a-f0-9]+):\n    span:\n      start: (?P<s>-?\d+)\n      end: (?P<e>-?\d+)",re.M)
split={r["book"]:r["split"] for r in csv.DictReader(open(ROOT/"Data/audit/split.csv"))}
PUNCT=re.compile(r"^[\s།༎༏༑༈]+")      # leading shad / spaces / newlines
PUNCT_R=re.compile(r"[\s།༎༏༑༈]+$")
TSHEG="་"; SHAD="།"
MAXN=3

def after_grams(t, pos):
    """1..3 syllables starting just after `pos` (punctuation skipped)."""
    seg = t[pos+1:pos+80]
    seg = PUNCT.sub("", seg)
    syl = [s for s in seg.split(TSHEG) if s][:MAXN+1]
    out=[]
    for k in range(1, min(MAXN,len(syl))+1):
        g=TSHEG.join(syl[:k])
        g=re.split(r"[\s།༎\n]", g)[0]
        if g: out.append(g)
    return out

def before_grams(t, pos):
    """1..3 syllables ending just before `pos`."""
    seg = t[max(0,pos-80):pos]
    seg = PUNCT_R.sub("", seg)
    syl = [s for s in seg.split(TSHEG) if s][-(MAXN+1):]
    out=[]
    for k in range(1, min(MAXN,len(syl))+1):
        g=TSHEG.join(syl[-k:])
        g=re.split(r"[\s།༎\n]", g)[-1]
        if g: out.append(g)
    return out

close_fg=collections.Counter(); close_bg=collections.Counter()
open_fg =collections.Counter(); open_bg =collections.Counter()
n_spans=0; n_bg=0
random.seed(11)

for bd in sorted(REPOS.iterdir()):
    if not bd.is_dir(): continue
    if split.get(bd.name)!="train": continue
    opf=bd/f"{bd.name}.opf"; base=opf/"base"/"v001.txt"; lay=opf/"layers"/"v001"/"Quotation.yml"
    if not (base.exists() and lay.exists()): continue
    t=base.read_text(encoding="utf-8",errors="replace")
    sp=[(int(m["s"]),int(m["e"])) for m in BLOCK.finditer(lay.read_text(encoding="utf-8",errors="replace"))]
    sp=[(s,e) for s,e in sp if e>=s]
    if not sp: continue
    ends={e for s,e in sp}; starts={s for s,e in sp}
    for s,e in sp:
        n_spans+=1
        for g in after_grams(t,e):  close_fg[g]+=1
        for g in before_grams(t,s): open_fg[g]+=1
    # background: shad positions far from any gold boundary
    shads=[m.start() for m in re.finditer(SHAD,t)]
    if not shads: continue
    random.shuffle(shads)
    take=0
    for p in shads:
        if take>=len(sp)*8: break
        if any(abs(p-x)<200 for x in ends) or any(abs(p-x)<200 for x in starts): continue
        take+=1; n_bg+=1
        for g in after_grams(t,p):  close_bg[g]+=1
        for g in before_grams(t,p): open_bg[g]+=1

json.dump(dict(n_spans=n_spans, n_bg=n_bg,
    close_fg=close_fg, close_bg=close_bg, open_fg=open_fg, open_bg=open_bg),
    open(Path.home()/"qd/mined.json","w"), ensure_ascii=False)
print("train spans:",n_spans," background points:",n_bg)
print("distinct closer grams:",len(close_fg)," opener grams:",len(open_fg))
