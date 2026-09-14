import json, collections, re
meta=json.load(open('qd/meta.json')); rows={r["book"]:r for r in json.load(open('qd/perbook.json'))}
SUF=re.compile(r"[–—-]\s*(Book|དེབ།).*$")
MERGE={"དིལ་མགོ་མཁྱེན་བཙེ་གསུང་འབུམ།":"Dilgo Khyentse Series","ཞབས་དཀར་གསུང་འབུམ།":"Shabkar Series",
       "ཤེས་བྱ་ཀུན་ཁྱབ་མཛོད།":"Treasury of Knowledge","ཐོས་བསམ་འཆད་ཉན་གྱི་རིམ་པ།":"RDI Study Series",
       "རིག་པའི་རྡོ་རྗེ་གཙུག་ལག་སློབ་ཉེར་ཁང་།":"Rigpe Dorje Institute"}
PMERGE={"Gautam Buddha ViharaDOWNLOAD EBOOK":"Gautam Buddha Vihara","རིག་པའི་རྡོ་རྗེ་གཙུག་ལག་སློབ་ཉེར་ཁང་།":"Rigpe Dorje Institute",
        "བེན་ཆེན་སློབ་གཉེར་ཁང་།":"Benchen Publications","Lekshey Ling Philosophy Series":"Lekshey Ling Publications"}
for b in meta:
    c=(meta[b]["collection"] or "").strip()
    s=re.sub(r"\s+"," ",SUF.sub("",c).strip(" –—-")) if c else "(none)"
    meta[b]["series"]=MERGE.get(s,s)
    p=meta[b]["publisher"] or "(none)"
    meta[b]["pub"]=PMERGE.get(p,p)
