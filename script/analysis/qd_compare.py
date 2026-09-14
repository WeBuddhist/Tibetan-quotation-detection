import csv, json, collections, statistics as st, random
from pathlib import Path
ROOT=Path.home()/"mnt"/"quotation_detection"
prof={r["book"]:r for r in csv.DictReader(open(ROOT/"Data/audit/quotation_tags/book_top3_profile.csv"))}
exec(open('qd/series.py').read())
per={r["book"]:r for r in json.load(open('qd/perbook.json'))}
K=["pct_text","pct_person","pct_formula","pct_chained","pct_misc"]
def ns(b): return int(prof[b]["n_new_sent"]) if b in prof else 0
BIG=[b for b in prof if ns(b)>=10]
SMALL=[b for b in per if b not in BIG and per[b]["Quotation_n"]>0]
ZERO=[b for b in per if per[b]["Quotation_n"]==0]
def dens(b): return per[b]["Quotation_chars"]/max(per[b]["doc_len"],1)
def bucket(b):
    x=dens(b)
    return 0 if per[b]["Quotation_n"]==0 else 1 if x<0.01 else 2 if x<0.03 else 3 if x<0.08 else 4 if x<0.15 else 5
def pool(sel):
    c=collections.Counter(); m=0
    for b in sel:
        if b not in prof: continue
        for k in K: c[k]+=float(prof[b][k])*ns(b)
        m+=ns(b)
    return ({k:100*c[k]/m for k in K} if m else {k:0 for k in K}), m
CORP,_=pool(BIG)
CD=100*sum(per[b]["Quotation_chars"] for b in per)/sum(per[b]["doc_len"] for b in per)
CPUB=collections.Counter()
for b in per: CPUB[meta.get(b,{}).get("pub","?")]+=per[b]["Quotation_n"]
MP=sum(CPUB.values())
def report(name, sel):
    p,m=pool(sel)
    d=100*sum(per[b]["Quotation_chars"] for b in sel)/max(sum(per[b]["doc_len"] for b in sel),1)
    pc=collections.Counter()
    for b in sel: pc[meta.get(b,{}).get("pub","?")]+=per[b]["Quotation_n"]
    tp=sum(pc.values()) or 1
    pdev=sum(abs(pc[k]/tp-CPUB[k]/MP) for k in CPUB)*100
    nz=sum(1 for b in sel if per[b]["Quotation_n"]==0)
    ser=len({meta.get(b,{}).get("series","?") for b in sel})
    bk=len({bucket(b) for b in sel})
    print(f"{name:26s} {len(sel):5d} {sum(per[b]['Quotation_n'] for b in sel):6d} "
          f"{p['pct_text']:6.1f} {p['pct_person']:7.1f} {p['pct_formula']:8.1f} {p['pct_chained']:8.1f} {p['pct_misc']:6.1f} "
          f"{d:7.2f} {pdev:7.1f} {len(pc):5d} {ser:6d} {bk:4d} {nz:4d}")
def greedy(objective, quota_frac=0.10, seed=0):
    random.seed(seed)
    sel=[]
    for bkt in sorted({bucket(b) for b in per}):          # DENSITY BUCKETS = hard strata
        cand=[b for b in per if bucket(b)==bkt]
        random.shuffle(cand)
        if bkt==0:                                        # zero-quotation books: proportional by count
            k=round(quota_frac*len(cand))
            sel+=cand[:k]; continue
        target=quota_frac*sum(per[b]["Quotation_n"] for b in cand); got=0
        while got<target and cand:
            best=None;bd=None
            for b in cand[:40]:
                d=objective(sel+[b])
                if bd is None or d<bd: bd,best=d,b
            sel.append(best); got+=per[best]["Quotation_n"]; cand.remove(best)
    return sel
def d_cat(s):
    p,_=pool(s); return sum(abs(p[k]-CORP[k]) for k in ("pct_text","pct_person","pct_misc"))
def d_pub(s):
    pc=collections.Counter()
    for b in s: pc[meta.get(b,{}).get("pub","?")]+=per[b]["Quotation_n"]
    t=sum(pc.values()) or 1
    return sum(abs(pc[k]/t-CPUB[k]/MP) for k in CPUB)*100
print(f"CORPUS   text {CORP['pct_text']:.1f}  person {CORP['pct_person']:.1f}  formula {CORP['pct_formula']:.1f} "
      f"chained {CORP['pct_chained']:.1f}  misc {CORP['pct_misc']:.1f}   density {CD:.2f}%   "
      f"{len(per)} books, {len({meta.get(b,{}).get('pub','?') for b in per})} publishers, "
      f"{len({meta.get(b,{}).get('series','?') for b in per})} series, {len(ZERO)} zero-quot books\n")
hdr=f"{'TEST SET':26s} {'books':>5s} {'spans':>6s} {'text':>6s} {'person':>7s} {'formula':>8s} {'chained':>8s} {'misc':>6s} {'dens':>7s} {'pubdev':>7s} {'pubs':>5s} {'series':>6s} {'bkt':>4s} {'0-q':>4s}"
print(hdr); print("-"*len(hdr))
report("current split", [b for b in per if per[b].get("split")=="test"])
report("publisher-stratified", greedy(d_pub, seed=1))
report("RECOMMENDED (cat+dens+pub)", greedy(lambda s: d_cat(s)+0.2*d_pub(s), seed=1))
random.seed(4)
report("random 24 books", random.sample(list(per),24))
