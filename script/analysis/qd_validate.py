"""Validate candidate opener/closer profiles. Derived on TRAIN, scored on VAL/TEST."""
import re, csv, json, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; REPOS=ROOT/"Data"/"repos"
BLOCK=re.compile(r"^  (?P<i>[a-f0-9]+):\n    span:\n      start: (?P<s>-?\d+)\n      end: (?P<e>-?\d+)",re.M)
split={r["book"]:r["split"] for r in csv.DictReader(open(ROOT/"Data/audit/split.csv"))}

TS="[་༌]"   # tsheg or non-breaking tsheg
Q=f"(?:ཞེས|ཅེས|ཤེས)"
CLOSERS={
 # existing, for comparison
 "v1_strong":  r"(?:ཞེས|ཅེས)་གསུངས|(?:ཞེས|ཅེས)་སོ|(?:ཞེས|ཅེས)་དང|གསུངས།",
 "v1_default": r"(?:ཞེས|ཅེས)་གསུངས|(?:ཞེས|ཅེས)་སོ|(?:ཞེས|ཅེས)་དང|གསུངས།|(?:ཞེས|ཅེས)་པ་(?:སྟེ|ལྟར|དང)",
 "v1_broad":   r"(?:ཞེས|ཅེས)(?:་|།)",
 # mined
 "v2_strong":  rf"(?:ཞེས|ཅེས){TS}(?:གསུངས|སོ|དང|བཤད|སོགས|པ{TS}(?:ལྟར|དང|སྟེ)|པས)|གསུངས{TS}?(?:སོ|པ|ནས)|(?<!་)གསུངས།",
 "v2_default": rf"(?:ཞེས|ཅེས)(?:{TS}(?:གསུངས|སོ|དང|བཤད|སོགས|པ|པས|ཞིང)|།|{TS}?$)|གསུངས",
 "v2_broad":   rf"(?:ཞེས|ཅེས)|གསུངས|བཤད{TS}པ",
}
OPENERS={
 "v1_draft":   r"ཇི་སྐད་དུ|ཇི་ལྟར|ཞལ་ནས|གསུང་ནས|གསུངས་པ་ལྟར|ལས།|ལས་ཀྱང|ཏུ།|དུ།|ན་རེ|ཞེས་བྱ་བ",
 "v2_strong":  rf"ཇི{TS}སྐད{TS}དུ|སྐད{TS}དུ|ཞལ{TS}ནས|གསུང{TS}ནས|ན{TS}རེ|(?:མདོ|རྒྱུད|མཛོད|འགྲེལ|རྒྱན|འཇུག|བསྟན{TS}བཅོས){TS}ལས|ལས{TS}ཀྱང",
 "v2_default": rf"ཇི{TS}སྐད{TS}དུ|སྐད{TS}དུ|ཞལ{TS}ནས|གསུང{TS}ནས|ན{TS}རེ|ལས{TS}?།|ལས{TS}ཀྱང|ལས$",
 "v2_broad":   rf"ལས|སྐད{TS}དུ|ཞལ{TS}ནས|ན{TS}རེ|ནས།",
}
CRE={k:re.compile(v) for k,v in CLOSERS.items()}
ORE={k:re.compile(v) for k,v in OPENERS.items()}

AFT_LO,AFT_HI = -30, 50     # closer search window around span end
BEF_LO,BEF_HI = -45, 10     # opener search window around span start

cov=collections.defaultdict(lambda: collections.Counter())   # (kind,prof,split) -> hits/total
prec=collections.defaultdict(lambda: [0,0])                  # (kind,prof) -> [at_boundary, total_occurrences]
DENSE=[]
books=[]
for bd in sorted(REPOS.iterdir()):
    if not bd.is_dir(): continue
    opf=bd/f"{bd.name}.opf"; base=opf/"base"/"v001.txt"; lay=opf/"layers"/"v001"/"Quotation.yml"
    if not (base.exists() and lay.exists()): continue
    t=base.read_text(encoding="utf-8",errors="replace")
    sp=[(int(m["s"]),int(m["e"])) for m in BLOCK.finditer(lay.read_text(encoding="utf-8",errors="replace"))]
    sp=[(s,e) for s,e in sp if e>=s]
    if not sp: continue
    books.append((bd.name, t, sp))
    qc=sum(e-s+1 for s,e in sp)
    if len(sp)>=100 and qc/len(t)>0.08: DENSE.append(bd.name)

for name,t,sp in books:
    sl=split.get(name,"?")
    for s,e in sp:
        tail=t[max(e+AFT_LO,0):e+AFT_HI]
        head=t[max(s+BEF_LO,0):s+BEF_HI]
        for p,rx in CRE.items():
            cov[("closer",p,sl)]["n"]+=1
            if rx.search(tail): cov[("closer",p,sl)]["hit"]+=1
        for p,rx in ORE.items():
            cov[("opener",p,sl)]["n"]+=1
            if rx.search(head): cov[("opener",p,sl)]["hit"]+=1

# precision proxy on densely annotated books only
for name,t,sp in books:
    if name not in DENSE: continue
    ends=sorted(e for s,e in sp); starts=sorted(s for s,e in sp)
    import bisect
    def near(pos, arr, lo, hi):
        i=bisect.bisect_left(arr,pos+lo)
        while i<len(arr) and arr[i]<=pos-lo:
            if lo<=pos-arr[i]<=hi or lo<=arr[i]-pos<=hi: return True
            i+=1
        return False
    for p,rx in CRE.items():
        for m in rx.finditer(t):
            prec[("closer",p)][1]+=1
            if any(-AFT_HI<=m.start()-e<=-AFT_LO or abs(m.start()-e)<=50 for e in ends[max(0,bisect.bisect_left(ends,m.start())-2):bisect.bisect_left(ends,m.start())+2]):
                prec[("closer",p)][0]+=1
    for p,rx in ORE.items():
        for m in rx.finditer(t):
            prec[("opener",p)][1]+=1
            if any(abs(m.end()-s)<=50 for s in starts[max(0,bisect.bisect_left(starts,m.end())-2):bisect.bisect_left(starts,m.end())+2]):
                prec[("opener",p)][0]+=1

print(f"dense books used for precision proxy: {len(DENSE)}\n")
for kind,profs in (("closer",CLOSERS),("opener",OPENERS)):
    print(f"=== {kind.upper()} coverage (% of gold spans matched) ===")
    print(f"{'profile':12s} {'train':>8s} {'val':>8s} {'test':>8s} {'ALL':>8s} | {'boundary-hit rate':>18s}")
    for p in profs:
        cells=[]
        tot_h=tot_n=0
        for sl in ("train","val","test"):
            c=cov[(kind,p,sl)]; tot_h+=c["hit"]; tot_n+=c["n"]
            cells.append(f"{100*c['hit']/max(c['n'],1):7.1f}%")
        a,b=prec[(kind,p)]
        print(f"{p:12s} "+" ".join(cells)+f" {100*tot_h/max(tot_n,1):7.1f}% | {100*a/max(b,1):16.1f}% ({b} occ)")
    print()
json.dump({"closers":CLOSERS,"openers":OPENERS}, open(Path.home()/"qd/profiles.json","w"), ensure_ascii=False, indent=2)
