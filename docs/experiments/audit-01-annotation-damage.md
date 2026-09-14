# audit-01 — Quotation answer-key damage

**Date:** 2026-08-18  **Status:** complete for Quotation, **not started for the other six layers**

## Question

eval-01 found 166,441 false-positive Quotation tokens originating from O. Are they all model
error, or is some share of them the model finding quotations nobody annotated?

## Method

Inspected annotation files directly for books with anomalously low quotation density, then
traced why the corpus audit had passed them.

## Results

Confirmed by reading the annotation files:

| book | chars | annotations | finding |
|---|---:|---|---|
| P000218 | 847,725 | 3 spans, all `start == end` | length-one stray marks — not quotations |
| P000271 | 443,169 | 1 span (46,778–46,878) | one real 101-char quotation in the whole book |
| P000078 | 842,034 | 15 real spans | 529,133-char gap between char 71,933 and 601,066 — **gap unverified** |

P000218's spans in full, from `Data/repos/P000218/P000218.opf/layers/v001/Quotation.yml`:

```yaml
a1f3e26bdb08449aaece0bba4ac4f5cc: {start: 16302, end: 16302}
670003b927a54a5aad937c6ae0206e61: {start: 16444, end: 16444}
4b95aa69bb6647d191e09a5e47dd5fa5: {start: 17717, end: 17717}
```

Flagged by density: **32 of 195** in-scope books below 0.5% quotation density against a
3.6% median. Only 15 were ever printed; the rest were computed in a session that no longer
exists. The suspect books skew large — several over 500,000 tokens — so they carry more
training signal than their count suggests. P000126 has **zero** spans while marked in scope.

## Why preprocessing didn't catch it

`classify_span` in `audit_utils.py` tests only geometric coherence:

```python
if start < 0 or end < 0:                     reasons.append("negative_offset")
if start > end:                              reasons.append("start_gt_end")
if doc_len >= 0 and (start >= doc_len or end >= doc_len):  reasons.append("oob")
if not reasons: return [], "keep", ...
```

`{16302, 16302}` is not negative, not inverted, not out of bounds — so it is kept. Under
inclusive-offset semantics, which the codebase defines as `end - start + 1`, it is a valid
one-character span. **There is no minimum-length rule anywhere in the pipeline.** `span_length()`
is used throughout the audit for reporting, and the only length-based rejection is
`long_quote_min`, an *upper* bound. Training's second gate, `normalize_spans`, rejects only
`end < start`, so the same spans pass twice.

The audit did drop 480 Quotation, 30 Chapter and 2 Yigchung spans — the genuinely broken
ones. 24,730 Quotation spans were kept.

**The zero-span book was detected and reported.** `Data/audit/clean_summary.csv` records
`books_in_scope` and `books_with_positive` per layer. Every layer matches — Sabche 138/138,
Tsawa 89/89, Chapter 159/159, Author 259/259, BookTitle 265/265 — except Quotation at
**195 / 194**. No assertion was ever placed on it.

## Findings

- The validator asks whether a span is geometrically *possible*. It was never asked whether a
  span is *plausible*. A one-character quotation is both coherent and absurd.
- Quote precision is understated by an unknown amount.
- Training received contradictory supervision — the same textual patterns annotated in one
  book and ignored in another.
- Validation scores partly reflect which books happened to land in validation.
- Every v1.2-vs-v1.3 comparison inherits all of this.

## What this changed about the plan

This became the binding constraint — ahead of the encoder, the learning rate and the loss
weights. Retraining, tokenizer work and the Tsawa-merge decision are all blocked behind it.

Two fixes are a few lines each: a per-layer minimum plausible span length in `classify_span`
(Author and Chapter spans are legitimately short, Quotation and Tsawa are not), and
`assert books_in_scope == books_with_positive`.

## Artifacts

[`FLAGGED_BOOKS.md`](../FLAGGED_BOOKS.md), `Data/audit/clean_summary.csv`,
`experiments/mmbert_base_v1.3/quote_fp_audit.json`.

## Caveats

**Quotation only.** Sabche, Tsawa, Yigchung, Chapter and Author have never been screened this
way. Given that four layers converged on nearly the same span F1 in eval-01, a shared
annotation ceiling is a live hypothesis — and untested. The flagged list is 15 of 32 books;
P000078's gap is flagged but unverified.
