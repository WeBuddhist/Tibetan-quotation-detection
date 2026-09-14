import yaml, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/"Data"/"repos"
out={}
for bd in sorted(ROOT.iterdir()):
    if not bd.is_dir(): continue
    m=bd/f"{bd.name}.opf"/"meta.yml"
    if not m.exists(): continue
    d=yaml.safe_load(m.read_text(encoding="utf-8"))
    w=(d.get("web_metadata") or {})
    s=(d.get("source_metadata") or {})
    def g(k,lang="en"):
        v=w.get(k)
        return (v.get(lang) or v.get("bo") or "").strip() if isinstance(v,dict) else (str(v).strip() if v else "")
    sku=str(w.get("sku") or s.get("id") or "")
    out[bd.name]={
      "publisher": g("pusblisher") or g("publisher"),
      "collection": g("collection"),
      "author": g("author") or (s.get("authors") or [""])[0],
      "category": g("category"),
      "org": g("org"),
      "sku": sku,
      "sku_prefix": "-".join(sku.split("-")[:2]) if sku else "",
      "sku_pub": sku.split("-")[0] if sku else "",
      "created": str(d.get("created_at") or ""),
      "declared_layers": s.get("layers") or [],
    }
json.dump(out, open(Path.home()/"qd/meta.json","w"))
print("books with meta:", len(out))
