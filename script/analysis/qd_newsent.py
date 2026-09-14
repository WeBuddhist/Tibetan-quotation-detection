"""Does the quotation start a NEW sentence (after །)? And what ended that previous sentence?"""
import re, csv, collections, random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; REPOS=ROOT/"Data"/"repos"
BLOCK=re.compile(r"^  (?P<i>[a-f0-9]+):\n    span:\n      start: (?P<s>-?\d+)\n      end: (?P<e>-?\d+)",re.M)
split={r["book"]:r["split"] for r in csv.DictReader(open(ROOT/"Data/audit/split.csv"))}
TSHEG=set("་༌"); SHAD=set("།༎༏༑༈༔༴༵༶"); WS=set(" \t\n\r ")
ERGX={"གིས","ཀྱིས","གྱིས","ཡིས"}
ERGN=re.compile(r"(?:པས|པོས|བས|ཆེས|ལྷས|ཧས|བོས|མས|རས|ཏིས|བུས|རྗེས|མཚོས|གྲགས|དཔལས|གེས|ཤས)$")
SPEECH={"གསུངས","སྩལ","སྨྲས","བཤད","བརྗོད","ཞུས","གསོལ","ཟེར"}
TERM={"དུ","ཏུ","རུ","སུ"}
def mark(o):
    if o in SPEECH or o in ERGX or ERGN.search(o): return "person"
    if o in ("ལས","ནས"): return "text"
    if o in TERM: return "formula"
    if o in ("ཞེས","ཅེས"): return "chained"
    return ""
def prev_syllable(t,s):
    """Returns (syllable, crossed_shad, crossed_space)."""
    i=s-1; sh=False; spc=False
    while i>=0 and (t[i] in WS or t[i] in SHAD):
        if t[i] in SHAD: sh=True
        if t[i] in WS: spc=True
        i-=1
    if i<0: return "",sh,spc
    j=i
    while j>=0 and t[j] not in TSHEG and t[j] not in SHAD and t[j] not in WS: j-=1
    return t[j+1:i+1], sh, spc
grp=collections.Counter(); marked=collections.Counter()
unmk=collections.defaultdict(collections.Counter); N=0
bysp=collections.defaultdict(collections.Counter)
random.seed(19); ex=[]
for bd in sorted(REPOS.iterdir()):
    if not bd.is_dir(): continue
    opf=bd/f"{bd.name}.opf"; base=opf/"base"/"v001.txt"; lay=opf/"layers"/"v001"/"Quotation.yml"
    if not (base.exists() and lay.exists()): continue
    t=base.read_text(encoding="utf-8",errors="replace"); sl=split.get(bd.name,"?")
    for m in BLOCK.finditer(lay.read_text(encoding="utf-8",errors="replace")):
        s,e=int(m["s"]),int(m["e"])
        if e<s: continue
        w,sh,spc=prev_syllable(t,s)
        g = "new sentence (after །)" if sh else "mid-sentence (no །)"
        N+=1; grp[g]+=1
        mk=mark(w)
        if mk: marked[(g,mk)]+=1
        else:
            unmk[g][w]+=1
            if sh and len(ex)<300: ex.append((bd.name,w,t[max(0,s-40):s].replace("\n","⏎"),t[s:s+30].replace("\n","⏎")))
        bysp[sl][(g, "marked" if mk else "unmarked")]+=1
print(f"spans={N}\n")
print("=== does the quotation start a new sentence? ===")
for g,c in grp.most_common():
    mk=sum(v for (gg,_),v in marked.items() if gg==g)
    print(f"  {g:24s} {c:6d} ({100*c/N:4.1f}% of spans)   marked: {mk:6d} = {100*mk/c:.1f}%")
print("\n=== of those, which marker type? ===")
print(f"{'group':24s} "+" ".join(f"{k:>9s}" for k in ("text","person","formula","chained")))
for g in grp:
    print(f"  {g:22s} "+" ".join(f"{100*marked[(g,k)]/grp[g]:8.1f}%" for k in ("text","person","formula","chained")))
print("\n=== NEW-SENTENCE spans with NO marker: what ended the previous sentence? ===")
g="new sentence (after །)"; tot=sum(unmk[g].values())
print(f"  ({tot} spans)")
for w,v in unmk[g].most_common(18):
    if len(w)<2: continue
    print(f"    {w:20s} {v:5d} {100*v/tot:5.2f}%")
print("\n=== by split: marked rate within each group ===")
print(f"{'split':6s} {'new-sent marked':>16s} {'mid-sent marked':>16s}")
for sl in ("train","val","test"):
    a=bysp[sl][("new sentence (after །)","marked")]; b=bysp[sl][("new sentence (after །)","unmarked")]
    c=bysp[sl][("mid-sentence (no །)","marked")]; d=bysp[sl][("mid-sentence (no །)","unmarked")]
    print(f"{sl:6s} {100*a/max(a+b,1):15.1f}% {100*c/max(c+d,1):15.1f}%")
print("\n=== samples: new sentence, previous sentence ended with no marker ===")
for bk,w,pre,h in random.sample(ex,6):
    print(f"  [{bk}] prev-final={w!r}\n     …{pre} ⟦{h}…")
