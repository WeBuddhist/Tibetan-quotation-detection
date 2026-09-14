# Suspected annotation damage — Quotation layer

**This list is INCOMPLETE.** 32 of 195 in-scope books have quote density below
0.5% (median is 3.6%). Only the 15 below were ever printed; the rest were computed
in a sandbox session that no longer exists. Regenerate with the script in §4.

## 1. Verified by reading the annotation file

| book | chars | annotations | finding | status |
|---|---:|---|---|---|
| **P000218** | 847,725 | 3 spans, all `start == end` | length-1 stray marks — **not quotations at all** | **confirmed damaged** |
| **P000271** | 443,169 | 1 span (46,778–46,878) | one real 101-char quotation in the whole book | **confirmed thin** |
| **P000078** | 842,034 | 15 real spans | 529,133-char gap between char 71,933 and 601,066 | **gap unverified** — see §3 |

P000218's three spans, in full:

```yaml
# Data/repos/P000218/P000218.opf/layers/v001/Quotation.yml
a1f3e26bdb08449aaece0bba4ac4f5cc: {start: 16302, end: 16302}
670003b927a54a5aad937c6ae0206e61: {start: 16444, end: 16444}
4b95aa69bb6647d191e09a5e47dd5fa5: {start: 17717, end: 17717}
```

These passed the corpus audit because it only rejects `start == end + 1`
(`inverted_empty_start_eq_end_plus_1`). `start == end` is technically valid.

## 2. Flagged by density only — NOT yet inspected

| book | tokens | Quote spans | density |
|---|---:|---:|---:|
| P000126 | 87,632 | **0** | 0.00000 |
| P000218 | 577,127 | 3 | 0.00001 |
| P000271 | 296,415 | 1 | 0.00025 |
| P000168 | 90,830 | 1 | 0.00059 |
| P000116 | 57,175 | 4 | 0.00059 |
| P000204 | 211,388 | 2 | 0.00080 |
| P000108 | 127,654 | 1 | 0.00085 |
| P000106 | 162,385 | 10 | 0.00112 |
| P000017 | 57,808 | 2 | 0.00112 |
| P000011 | 446,041 | 43 | 0.00187 |
| P000169 | 474,758 | — | 0.00316 |
| P000171 | 639,460 | — | 0.00355 |
| P000129 | 324,985 | — | 0.00421 |
| P000078 | 566,614 | 15 | 0.00454 |
| P000112 | 249,181 | — | 0.00478 |

**15 of 32.** The suspect books skew large (several over 500k tokens), so they
carry more training signal than their count suggests.

## 3. What is NOT established

- **The 17 unlisted books.** Never printed.
- **P000078's gap.** I flagged a 529k-char stretch with no annotations but never
  checked whether it contains quotable material. If the book's 555 quotation
  closers cluster in the annotated regions, there is no gap problem and it should
  come off this list.
- **Whether low density means damage.** Some works genuinely quote little. Only
  P000218 is beyond interpretation, because a one-character quotation cannot be a
  judgement call.

## 4. Where everything lives

```
Data/repos/<BOOK>/<BOOK>.opf/layers/v001/Quotation.yml   annotation spans
Data/repos/<BOOK>/<BOOK>.opf/base/v001.txt               raw text
Data/repos/<BOOK>/<BOOK>.opf/meta.yml                    layer declarations
Data/output/tokenized/<BOOK>.json                        offsets + BIO labels
Data/audit/flagged_spans.csv                             existing audit (different problems)
Data/audit/split.csv                                     train/val/test assignment
```

Regenerate the full list (needs the sandbox back, or run locally):

```python
import json, glob
rows = []
for p in glob.glob("Data/output/tokenized/*.json"):
    d = json.load(open(p))
    if not d.get("in_scope", {}).get("Quotation"):
        continue
    n = d["n_tokens"]
    q = d["label_counts"].get("B-QUOTE", 0) + d["label_counts"].get("I-QUOTE", 0)
    if n:
        rows.append((q / n, d["book"], d["span_counts"].get("Quotation", 0), n))
for dens, book, spans, n in sorted(rows):
    if dens < 0.005:
        print(f"{book}  density {dens:.5f}  spans {spans:4d}  tokens {n:>9,}")
```

## 5. The screen worth building

Density alone produces false alarms. Three signals together are far more specific:

1. **Degenerate spans** — `start == end`, or length below ~5 characters. Catches
   P000218 with certainty and needs no interpretation.
2. **Positional gaps** — annotations stop for a long stretch while quotation
   closers (ཞེས་གསུངས / ཅེས་གསུངས / ཞེས་སོ / ཅེས་སོ) continue. This is what would
   confirm or clear P000078, and it's the generalisable version of that check.
3. **Density outliers** — the list in §2, used to rank rather than to conclude.

Run across all 195 in-scope books, that turns "we found three bad books" into a
number: what fraction of the Quotation training signal is unreliable. That number
decides whether you clean the data, filter the corpus, or train treating
unlabelled tokens as unknown rather than negative.
