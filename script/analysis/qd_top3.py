"""Final: top-3 opener categories + misc, over NEW-SENTENCE spans."""
import re, csv, collections, statistics as st, math, random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; REPOS=ROOT/"Data"/"repos"; OUT=ROOT/"Data"/"audit"/"quotation_tags"
BLOCK=re.compile(r"^  (?P<i>[a-f0-9]+):\n    span:\n      start: (?P<s>-?\d+)\n      end: (?P<e>-?\d+)",re.M)
split={r["book"]:r["split"] for r in csv.DictReader(open(ROOT/"Data/audit/split.csv"))}
TSHEG=set("་༌"); SHAD=set("།༎༏༑༈༔༴༵༶"); WS=set(" \t\n\r ")
ERGX={"གིས","ཀྱིས","གྱིས","ཡིས"}
ERGN=re.compile(r"(?:པས|པོས|བས|ཆེས|ལྷས|ཧས|བོས|མས|རས|ཏིས|བུས|རྗེས|མཚོས|གྲགས|དཔལས|གེས|ཤས)$")
SPEECH={"གསུངས","སྩལ","སྨྲས","བཤད","བརྗོད","ཞུས","གསོལ","ཟེར"}; TERM={"དུ","ཏུ","རུ","སུ"}
def classify(w,p):
    if w=="དང" and p in ("ཞེས","ཅེས"): return "CHAINED"
    if w in ("ཀྱང","ཡང") and p: w=p                  # read over ཀྱང/ཡང
    if w in ("ལས","ནས"):
        return "PERSON" if p in ("ཞལ","གསུང") else "TEXT"
    if w in SPEECH or w in ERGX or ERGN.search(w): return "PERSON"
    if w in TERM: return "FORMULA"
    if w in ("ཞེས","ཅེས"): return "CHAINED"
    return "MISC"
def prev_two(t,s):
    i=s-1; sh=False
    while i>=0 and (t[i] in WS or t[i] in SHAD or t[i] in TSHEG):
        if t[i] in SHAD: sh=True
        i-=1
    out=[]
    for _ in range(2):
        if i<0: break
        j=i
        while j>=0 and t[j] not in TSHEG and t[j] not in SHAD and t[j] not in WS: j-=1
        out.append(t[j+1:i+1]); i=j-1
        while i>=0 and (t[i] in WS or t[i] in SHAD or t[i] in TSHEG): i-=1
    return (out+["",""])[:2], sh
CL=["TEXT","PERSON","FORMULA","CHAINED","MISC"]
tot=collections.Counter(); bysp=collections.defaultdict(collections.Counter)
misc=collections.Counter(); lens=collections.defaultdict(list); closer=collections.defaultdict(collections.Counter)
book=collections.defaultdict(collections.Counter); bs={}; N=0
def syl(x): return [p for p in re.split("["+re.escape("་༌ \n\r།༎༏༑༈༔")+"]+",x) if p]
for bd in sorted(REPOS.iterdir()):
    if not bd.is_dir(): continue
    opf=bd/f"{bd.name}.opf"; base=opf/"base"/"v001.txt"; lay=opf/"layers"/"v001"/"Quotation.yml"
    if not (base.exists() and lay.exists()): continue
    t=base.read_text(encoding="utf-8",errors="replace"); sl=split.get(bd.name,"?")
    sp=[(int(m["s"]),int(m["e"])) for m in BLOCK.finditer(lay.read_text(encoding="utf-8",errors="replace"))]
    sp=[(s,e) for s,e in sp if e>=s]
    if not sp: continue
    bs[bd.name]=sl
    for s,e in sp:
        (w,p),sh=prev_two(t,s)
        if not sh: continue                # NEW-SENTENCE spans only
        k=classify(w,p); N+=1
        tot[k]+=1; bysp[sl][k]+=1; book[bd.name][k]+=1
        lens[k].append(e-s+1)
        if k=="MISC": misc[w]+=1
        nx=syl(t[e+1:e+40])
        closer[k]["Q" if nx and nx[0] in ("ཞེས","ཅེས") else "G" if nx and nx[0]=="གསུངས" else "-"]+=1
print(f"new-sentence quotation spans: {N}\n")
print("=== THE CATEGORIES ===")
print(f"{'#':2s} {'category':9s} {'meaning':26s} {'spans':>6s} {'%':>7s} {'med len':>8s} {'closed':>7s}")
NAMES={"TEXT":"from a text (ལས/ནས)","PERSON":"from a person (-ས/ཞལ་ནས)","FORMULA":"as it is said (སྐད་དུ)","CHAINED":"…and (ཞེས་དང)","MISC":"no marker"}
for i,k in enumerate(CL,1):
    c=tot[k]; cc=closer[k]; n=sum(cc.values())
    print(f"{i:<2d} {k:9s} {NAMES[k]:26s} {c:6d} {100*c/N:6.1f}% {st.median(lens[k]):8.0f} {100*cc['Q']/max(n,1):6.0f}%")
print("\n=== by split ===")
print(f"{'split':6s} "+" ".join(f"{k:>9s}" for k in CL))
for sl in ("train","val","test"):
    n=sum(bysp[sl].values()); print(f"{sl:6s} "+" ".join(f"{100*bysp[sl][k]/n:8.1f}%" for k in CL))
print("\n=== what's in MISC ===")
m=sum(misc.values())
for w,v in misc.most_common(10):
    if len(w)>=2: print(f"    {w:14s} {v:5d} {100*v/m:5.1f}% of misc")
ok=[b for b in book if sum(book[b].values())>=10]
T=collections.Counter()
for b in ok: T.update(book[b])
NT=sum(T.values())
print(f"\n=== book separation ({len(ok)} books) ===")
for k in CL:
    p=T[k]/NT; sh=[book[b][k]/sum(book[b].values()) for b in ok]; ns=[sum(book[b].values()) for b in ok]
    sd=st.pstdev(sh); exp=st.mean([math.sqrt(p*(1-p)/n) for n in ns])
    print(f"  {k:9s} {100*p:6.1f}%   {sd/exp:5.1f}x above chance")
rows=[dict(book=b, split=bs[b], n_new_sent=sum(book[b].values()),
      **{f"pct_{k.lower()}":round(book[b][k]/sum(book[b].values()),4) for k in CL}) for b in book]
with open(OUT/"book_top3_profile.csv","w",newline="",encoding="utf-8") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
random.seed(1)
def share(sel,c="TEXT"):
    cc=collections.Counter()
    for b in sel: cc.update(book[b])
    return 100*cc[c]/sum(cc.values())
r=sorted(share(random.sample(ok,24)) for _ in range(3000))
print(f"\n=== random 24-book test sets, TEXT share: mean {st.mean(r):.1f}% sd {st.pstdev(r):.1f} "
      f"5-95% [{r[150]:.1f}, {r[2850]:.1f}]  |  actual test {share([b for b in ok if bs[b]=='test']):.1f}%")
print(f"wrote {OUT/'book_top3_profile.csv'}")
