"""Final sweep: quote SOURCE TYPE from རྣམ་དབྱེ་, clean-boundary spans only."""
import re, csv, collections, statistics as st, math, random
from pathlib import Path
ROOT=Path.home()/"mnt"/"quotation_detection"; REPOS=ROOT/"Data"/"repos"; OUT=ROOT/"Data"/"audit"/"quotation_tags"
BLOCK=re.compile(r"^  (?P<i>[a-f0-9]+):\n    span:\n      start: (?P<s>-?\d+)\n      end: (?P<e>-?\d+)",re.M)
split={r["book"]:r["split"] for r in csv.DictReader(open(ROOT/"Data/audit/split.csv"))}
DELIM="་༌ \n\r།༎༏༑༈༔༴༵༶྅"; SP=re.compile("["+re.escape(DELIM)+"]+"); PUN=set(DELIM); TS="་"
def syl(x): return [p for p in SP.split(x) if p]
ERGX={"གིས","ཀྱིས","གྱིས","ཡིས"}
ERGN=re.compile(r"(?:པས|པོས|བས|ཆེས|ལྷས|ཧས|བོས|མས|རས|ཏིས|བུས|རྗེས|མཚོས|གྲགས|དཔལས|གེས|ཤས|ལུས|ཁས)$")
SPEECH={"གསུངས","སྩལ","སྨྲས","བཤད","བརྗོད","ཞུས","གསོལ","ཟེར"}
TERM={"དུ","ཏུ","རུ","སུ"}
SKIP={"ཀྱང","ཡང"}
def source_type(pb):
    """Who/what is the quote from? Reads back over ཀྱང/ཡང."""
    i=len(pb)-1
    while i>=0 and pb[i] in SKIP: i-=1
    if i<0: return "none"
    o=pb[i]; prev=pb[i-1] if i>0 else ""
    if o=="ནས" and prev in ("ཞལ","གསུང","ཞལ་ནས"): return "person"
    if o in ("ན་རེ",) or o in SPEECH: return "person"
    if o in ERGX or ERGN.search(o): return "person"
    if o in ("ལས","ནས"): return "text"
    if o in TERM: return "formula"
    if o in ("ཞེས","ཅེས") or (prev in ("ཞེས","ཅེས") and o=="དང"): return "chained"
    return "none"
CL=["text","person","formula","chained","none"]
book=collections.defaultdict(collections.Counter); bs={}
tot=collections.Counter(); bysp=collections.defaultdict(collections.Counter)
lens=collections.defaultdict(list); closer=collections.defaultdict(collections.Counter)
N=0
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
        if not (s==0 or t[s-1] in PUN): continue          # CLEAN start only
        k=source_type(syl(t[max(0,s-80):s]))
        book[bd.name][k]+=1; tot[k]+=1; bysp[sl][k]+=1; N+=1
        lens[k].append(e-s+1)
        nx=syl(re.sub(r"^[\s།༎༏༑༈༔\n\r]+","",t[e+1:e+40])[:20])
        closer[k]["quotative" if nx and nx[0] in ("ཞེས","ཅེས") else "གསུངས" if nx and nx[0]=="གསུངས" else "none"]+=1
print(f"clean-start spans: {N}\n")
print("=== 1. quote SOURCE type ===")
print(f"{'type':9s} {'spans':>6s} {'%':>7s} {'median len':>11s} {'closed by ཞེས/ཅེས':>19s}")
for k in CL:
    c=tot[k]; cc=closer[k]; n=sum(cc.values())
    print(f"{k:9s} {c:6d} {100*c/N:6.1f}% {st.median(lens[k]):11.0f} {100*cc['quotative']/n:18.0f}%")
print("\n=== 2. by split ===")
print(f"{'split':6s} "+" ".join(f"{c:>9s}" for c in CL))
for s in ("train","val","test"):
    n=sum(bysp[s].values()); print(f"{s:6s} "+" ".join(f"{100*bysp[s][c]/n:8.1f}%" for c in CL))
ok=[b for b in book if sum(book[b].values())>=10]
T=collections.Counter()
for b in ok: T.update(book[b])
NT=sum(T.values())
print(f"\n=== 3. do books differ? ({len(ok)} books) ===")
print(f"{'type':9s} {'corpus':>8s} {'book sd':>9s} {'chance':>8s} {'ratio':>7s}")
for k in CL:
    p=T[k]/NT; sh=[book[b][k]/sum(book[b].values()) for b in ok]; ns=[sum(book[b].values()) for b in ok]
    sd=st.pstdev(sh); exp=st.mean([math.sqrt(p*(1-p)/n) for n in ns])
    print(f"{k:9s} {100*p:7.1f}% {sd:9.3f} {exp:8.3f} {sd/exp:6.1f}x")
random.seed(1)
def share(sel,c="text"):
    cc=collections.Counter()
    for b in sel: cc.update(book[b])
    return 100*cc[c]/sum(cc.values())
rnd=[share(random.sample(ok,24)) for _ in range(3000)]
act=share([b for b in ok if bs[b]=="test"])
r=sorted(rnd)
print(f"\n=== 4. random 24-book test sets: TEXT-citation share ===")
print(f"  mean {st.mean(r):.1f}%  sd {st.pstdev(r):.1f}  5-95% [{r[150]:.1f}, {r[2850]:.1f}]   actual test {act:.1f}%")
rows=[dict(book=b, split=bs[b], n_clean=sum(book[b].values()),
      **{f"src_{c}":round(book[b][c]/sum(book[b].values()),4) for c in CL}) for b in book]
with open(OUT/"book_source_type.csv","w",newline="",encoding="utf-8") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"\nwrote {OUT/'book_source_type.csv'}")
