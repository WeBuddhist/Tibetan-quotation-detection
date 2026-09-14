# Handoff — Tibetan Quotation Benchmark: split, leakage audit, Hugging Face release

**Repo:** `~/Developer/quotation_detection`
**Everything below goes in ONE new module.** Do not spread this across helper files.

- Module: `src/layer_detection/quotation_benchmark.py`
- CLI wrapper: `script/build_quotation_benchmark.py` (six lines — import `main`, call it)

Do **not** touch `split.py`, `split_utils.py`, or `dataset_utils.py`. Those drive the old
7-layer experiment and must keep working unchanged. The quotation benchmark is a separate
experiment and lives on its own.

---

## 1. What this is

We have a reviewed, human-approved gold set of Tibetan quotation annotations. We need to:

1. Split it into train / val / test so the three splits look alike.
2. Measure and write down how much text leaks between them.
3. Push it to the Hugging Face Hub with a dataset card that explains both honestly.

---

## 2. Input — this file already exists, do not regenerate it

`Data/audit/quotation_gold.json`

```json
{
  "generated": "2026-09-09T15:50:52+00:00",
  "policy": "book-level: an accepted book contributes every span; a rejected book contributes none",
  "kept_books": 320,
  "kept_spans": 28892,
  "dropped_spans": 17163,
  "snapped_spans": 254,
  "class_counts": {"TEXT": 13543, "MISC": 9061, "PERSON": 4181, "FORMULA": 1138, "CHAINED": 969},
  "books": {
    "IB522F095": {
      "layer": "Citation",          // "Citation" (new batch) or "Quotation" (old batch) — same annotation
      "batch": "new",               // "old" | "new"
      "verdict": "accept",
      "title": "...",
      "doc_chars": 281185,          // length of base/v001.txt
      "closed_pct": 19.4,
      "misaligned_pct": 0.0,
      "both_mid_pct": 2.0,
      "line_split_pct": 71.9,
      "n_original": 1211,
      "spans": [
        {"ann": "fd5736cb...", "start": 3850, "end": 3975, "cls": "MISC", "closed": false}
      ]
    }
  }
}
```

Two things to know about the spans:

- **Offsets are inclusive on both ends.** Span length is `end - start + 1`. The character at
  `end` is part of the quotation. Getting this wrong shifts every label by one.
- **`cls` is already computed.** It is the marker class from the human review pass
  (`TEXT` / `PERSON` / `MISC` / `FORMULA` / `CHAINED`). Do not recompute it and do not import
  `quote_markers.py` here. Read the value that is in the file.

Raw text for book `B` is at `Data/repos/<B>/<B>.opf/base/v001.txt`.

Books in the gold file are the 320 that a human accepted. The other 78 reviewed books were
rejected whole and are already gone. Never add them back.

### 2.1 The spans are chunked — read this before you write any code

A long quotation in the new batch is **not stored as one span**. It is cut into pieces of
roughly 20–40 or 110–160 characters, split at newlines, with each piece stored as its own
annotation. 7,207 spans (24.9%) are pieces of a longer quotation. They are not one span per
verse line — the pieces themselves contain 8,658 newlines, so the cut points are arbitrary.

This is almost entirely a new-batch artifact: 28.7% of new-batch spans continue the previous
span, against 0.7% in the old batch.

It matters more than it looks, because it corrupts the class labels. `cls` is decided by the
marker in front of a span. Only the *first* piece of a chunked quotation has that marker; the
rest have nothing in front of them and fall through to `MISC`. The damage is plain in the
counts:

| | TEXT | PERSON | MISC |
|---|---|---|---|
| first piece of a chain | 793 | 220 | 540 |
| continuation pieces | 166 | 517 | 4,694 |

So a large part of the `MISC` class is not a real category — it is continuation text with the
marker cut off. **Merge before you do anything else, or you will stratify on a fiction.**

---

## 3. The split

### 3.0 Step one: merge continued spans

Walk each book's spans in start order. Join span A to the next span B when both hold:

1. The gap `B.start - A.end - 1` is 0, 1 or 2 characters, and the text in that gap is only a
   newline, a space, a shad (`།`), or nothing.
2. **A does not already end on a closer.** Test A's own text against
   `(ཞེས|ཅེས|ཤེས|གསུངས|འབྱུང|བཤད|སྨྲས)[^།]{0,12}།{1,2}\s*$`.

The merged span runs `first.start` to `last.end`, and takes the **class of the first piece** —
that is the one whose marker was actually read.

Rule 2 is not optional. Separate quotations often sit on consecutive lines, each closed with
`ཞེས་སོ།།`, and a naive merge welds them into one. The guard blocks 115 of 5,531 candidate
joins (2.1%). Real case from `IB522F095`, three back-to-back quotations that must stay apart:

```
...ཆེན་པོ་ཡིན་པར་ཤེས་པར་བྱ།། ཞེས་སོ།།          <- ends closed, do not join
ཡང་བཤེས་གཉེན་དུལ་བ་ཞི་བ་... ཞེས་སོ།། ཡང་།      <- ends closed, do not join
རྒྱུད་འཇམ་ཞེ་སྡང་རབ་སྤངས་ཤིང་།། ... ཞེས་སོ།།
```

And one that must be joined — same book, one quotation cut into three:

```
བྱང་ཆུབ་སེམས་དང་སྡོམ་པ་དང་།། ... དངོས་པོ་དང་།།
བགྱིད་དང་བགྱིད་དུ་སྩལ་བ་དང་།། ... སྐྱེས་པ་དང་།།
རང་གཞན་འཁོར་འདས་ཐམས་ཅད་ཀྱིས།། ... བསྔོ།། གསུངས་སོ།།
```

Result: **28,892 spans become 23,476.** Everything downstream — targets, split, export — uses
the merged spans. Write the merged set to `Data/audit/quotation_gold_merged.json` in the same
shape as the input, with a `merged_from` count on each span, and leave the original file alone.

### 3.0b Step two: drop the under-annotated books

Some books in the gold set carry almost no annotations across hundreds of thousands of
characters. They are not books that rarely quote — they are books nobody finished annotating.
The module must prove this rather than assume it, then act on it.

**The diagnostic.** For every book, count quotation closer markers in the raw text with
`(ཞེས|ཅེས)[^།]{0,20}(གསུངས|བཤད|འབྱུང|སོ|པ|པའི)`, then count how many of those markers have an
annotated span ending within 30 characters before them. Group books by density and print:

| book density | books | chars | spans | closers | closers per Mchar | closers with an annotation |
|---|---|---|---|---|---|---|
| 0–0.25% | 21 | 4.7M | 45 | 2,707 | 578 | **1.1%** |
| 0.25–0.5% | 24 | 5.7M | 148 | 6,941 | 1,208 | 1.2% |
| 0.5–1% | 21 | 4.8M | 261 | 4,772 | 1,000 | 3.5% |
| 1–3% | 78 | 17.3M | 2,542 | 20,982 | 1,210 | 7.7% |
| 3–8% | 79 | 18.4M | 7,526 | 21,849 | 1,190 | 19.0% |
| 8–15% | 55 | 10.9M | 9,511 | 14,053 | 1,289 | 35.8% |
| ≥15% | 42 | 5.6M | 8,859 | 7,268 | 1,292 | **45.6%** |

The marker rate is flat — about 1,200 per million characters in every band. Sparse books quote
as often as dense ones. Only the annotation changes: 1.1% coverage at the bottom, 45.6% at the
top. The regex over-fires a little, but that error hits every row equally, so the 40× ratio is
the finding, not the absolute counts.

**The action.** Drop books whose merged density is below **1.0%**. That removes 66 books and
22.5% of the text while losing 377 spans — 1.6% of the annotations — and moves the class mix
by one tenth of a point. Below 1% the marker coverage is 1–3%, which is indistinguishable from
unannotated; above it, coverage climbs steadily.

Why this matters more than tidiness: these books are false negatives. In test they punish the
model for being right — it finds a real quotation, the gold has nothing there, and it scores
as a false positive. In training they teach it to stay quiet. And they are not usable as clean
negatives either, because they contain real unmarked quotations.

Make the threshold a `--min-density` flag defaulting to 1.0. Write the full diagnostic table to
`Data/audit/quotation_annotation_coverage.md` and name the dropped books there.

**Result after §3.0 and §3.0b: 254 books, 23,099 spans, density 7.41%.**

Note that 7.41% is the density *of adequately annotated books*, not of a random Tibetan book.
The dataset card must say which one it is. Also be honest that the cut does not fix everything
— the 1–3% band still sits at 7.7% marker coverage. That belongs in the card's known limits,
with this table as the evidence.

### 3.1 Book-level, always

A book goes entirely into one split. Never split a book's spans across train/val/test.
This is the single most important rule — see the leakage section for why.

### 3.2 What we balance on

The three splits must match the whole corpus on these four things, in this order of
importance:

All targets below are measured on the **merged, density-filtered** set from §3.0 and §3.0b —
254 books, 23,099 spans. Recompute them from the data, never hardcode them.

| # | What | Weight | Corpus target |
|---|---|---|---|
| 1 | Marker class mix, over **TEXT / PERSON / MISC only** | 1.0 | 57.0% / 15.7% / 18.9% |
| 2 | Density-bucket mix (see below) | 0.5 | 0 / 9.2 / 26.1 / 32.9 / 31.8% |
| 3 | Batch mix (share of spans from `old` books) | 0.5 | 39.8% |
| 4 | Overall quotation density | 20.0 | 7.41% |

The `<1%` bucket is empty by construction after §3.0b, so only four buckets carry weight.
Keep the five-bucket code anyway — it should fall out of the data, not be special-cased.

**Yes — the class balancing is TEXT, PERSON and MISC, and only those three.** This was
tested on the unmerged set: balancing on TEXT alone let MISC drift 7 points off target, adding
PERSON and MISC cut the total five-class error from 14.20 to 4.40, and also pinning FORMULA
and CHAINED bought only 0.66 more. Those two are about 4% of spans each, so the search starts
trading away density and batch balance to chase roughly 100 spans. Leave them free and expect
them to land about half a point high. That is the accepted cost and it goes in the dataset
card. Re-run that three-way comparison on the merged spans and print it — the conclusion
should hold, but confirm rather than assume.

Density of a book = `100 * (sum of span lengths) / doc_chars`.

Density buckets: `<1%`, `1–3%`, `3–8%`, `8–15%`, `>=15%`. Bucket share is measured in
**spans**, not books.

### 3.3 The algorithm

Greedy, with test and val built **at the same time** — not one after the other.

```
targets   = {"test": 0.10 * total_spans, "val": 0.10 * total_spans}
candidates = all books, sorted by span count, LARGEST FIRST
book cap  = 40 per held-out split

loop:
    hungry = splits that are still under their span quota and under the book cap
    if none: stop
    window = first 60 unused candidates (first 120 once a split is 60% full by book count)
    for each hungry split s, for each book b in window:
        delta = cost(s + b) - cost(s)
    take the (s, b) pair with the smallest delta
train = everything left over
```

`cost(split)` is the weighted sum of absolute percentage-point gaps from §3.2:

```
cost = 1.0 * sum(|pct[c] - target[c]| for c in (TEXT, PERSON, MISC))
     + 0.5 * sum(|bucket_pct[k] - bucket_target[k]| for k in 0..4)
     + 0.5 * |old_pct - 32.9|
     + 20.0 * |density - 5.82|
```

Two details that matter and cost hours if you get them wrong:

- **Largest book first.** A plain greedy search prefers small books because they let it
  fine-tune the percentages cheaply. The result is a test set full of tiny books whose
  density collapses to 3.40% against a 5.82% corpus. Sorting largest-first plus the book cap
  is what fixes it.
- **Build test and val jointly.** Building test first and then val from the leftovers gives a
  good test set and a poor val set (profile error 3.11, MISC at 29.0%). Joint construction
  brings both under 1.0.

Accumulate class counts, bucket counts, span counts, quotation chars, doc chars and old-batch
counts incrementally per candidate. Recomputing from scratch inside the inner loop makes this
run for minutes instead of seconds.

### 3.4 Sweep and pick

Do not hardcode a config. Sweep `density_weight` in `(10, 20, 30)`, `batch_weight` in
`(0.5, 1.5)`, `book_cap` in `(25, 30, 40)`, and **restart seed in `range(12)`**, and pick the
one that minimises:

```
profile_err(test) + profile_err(val)
  + 1.5 * (|density(test) - corpus| + |density(val) - corpus|)
  + 0.5 * (|old_pct(test) - corpus| + |old_pct(val) - corpus|)
```

where `profile_err` is the sum of absolute gaps on TEXT/PERSON/MISC.

The restart seed matters. Plain greedy is order-dependent, and after the §3.0b cut the pool is
254 books instead of 320 — small enough that it gets stuck. With a fixed order the best test
profile error I could reach was 2.37; letting the seed break ties among same-size books and
shuffle the candidate window brought it to 1.19. The seed only permutes the order; it never
changes the objective. Record the winning seed in the output so the run reproduces exactly.

Print the whole sweep table, then the winning split. No unseeded randomness anywhere.

### 3.5 What good looks like

Reference run, merged and density-filtered (density weight 20, batch weight 0.5, cap 30,
seed 4):

| split | books | spans | TEXT | PERSON | MISC | density | old% | profile err |
|---|---|---|---|---|---|---|---|---|
| train | 211 | 18,428 | 57.0% | 15.8% | 18.9% | 7.40% | 40.3% | 0.14 |
| val | 25 | 2,346 | 57.1% | 15.6% | 18.9% | 7.51% | 38.4% | 0.24 |
| test | 18 | 2,325 | 56.4% | 15.7% | 18.3% | 7.38% | 37.2% | 1.19 |
| **corpus** | **254** | **23,099** | **57.0%** | **15.7%** | **18.9%** | **7.41%** | **39.8%** | — |

Anything with test and val profile error under about 1.5 and density within 0.15 points of
7.41% is fine. Batch is the loosest of the four here — test at 37.2% against 39.8% is the
known soft spot, and a wider sweep may beat it. If your winner misses any target, say so in
the output rather than hiding it.

### 3.6 Split outputs

- `Data/audit/quotation_split.csv` — one row per book: `book, split, batch, layer, n_spans,
  doc_chars, quote_chars, density, bucket, pct_TEXT, pct_PERSON, pct_MISC, pct_FORMULA,
  pct_CHAINED, title`
- `Data/audit/quotation_split.json` — `{book: split}`, plus a `_meta` key holding the chosen
  weights, the corpus targets, and the per-split measured table. This is what makes the run
  reproducible.

---

## 4. Leakage — measure it, do not pretend it is zero

There are two different things people mean by leakage here. Keep them apart in both the code
and the card.

### 4.1 Structural leakage — we prevent this completely

Because the split is book-level, no character of any book is in two splits. No span, no
window, no sentence, no overlapping context. This one is genuinely zero, and the module must
assert it: check that the three book sets are disjoint and that their union is exactly the
320 gold books. Fail loudly if not.

### 4.2 Quotation-text overlap — real, unavoidable, must be documented

Tibetan commentaries quote the same canonical scriptures over and over. So the *same
quotation string* shows up in books that land in different splits. Measured after stripping
whitespace, tsheg (`་`, `༌`) and shad (`།`, `༑`, `༔`).

**The table below was measured on the unmerged spans.** Recompute it on the merged spans and
publish those figures instead — merging makes spans longer, and longer strings match by
coincidence less often, so expect the overlap rate to move. Treat these as the order of
magnitude, not the number to quote:

| min length | spans | in more than one book | text also crosses a split boundary | held-out spans whose exact text is also in train |
|---|---|---|---|---|
| 10 chars | 28,862 | 4,157 (14.4%) | 1,680 (5.8%) | **690 / 5,874 (11.7%)** |
| 20 chars | 27,165 | 3,965 (14.6%) | 1,649 (6.1%) | **675 / 5,578 (12.1%)** |
| 30 chars | 23,960 | 3,654 (15.3%) | 1,543 (6.4%) | **630 / 4,874 (12.9%)** |

So about **12% of held-out spans quote a passage that also appears somewhere in training**.

This is not a bug in the split and you should not try to fix it by moving books around. It is
what the domain is. Forcing zero overlap would mean throwing out most of the corpus, and it
would also make the benchmark unrepresentative — a real model deployed on Tibetan texts *will*
meet quotations it has seen before. Deduplicating would measure a task nobody has.

What we do instead:

1. **Measure it and publish the number.**
2. **Tag every held-out span** with a boolean `seen_in_train` — true when its normalised text
   appears in any training book. Put this field in the released dataset.
3. **Require two headline scores** in every eval from now on: overall, and the `seen_in_train
   == false` subset. The gap between them is the memorisation effect, out in the open.

The module writes `Data/audit/quotation_leakage.md` — the table above, recomputed for the
actual chosen split, plus the per-split counts of `seen_in_train`.

Use the same normalisation everywhere: strip `[\s་༌།༑༔]+`. Compare on exact normalised string
match at a 20-character minimum for the headline figure (shorter strings match by accident).

### 4.3 Book duplication

Before anything else, check whether any two book IDs hold the same base text (some OpenPecha
books were ingested twice). Hash `base/v001.txt` per book. If duplicates exist, keep the one
with more spans, drop the other from the gold set entirely, and report it. Do not let a
duplicate pair straddle a split — that would be true structural leakage wearing a disguise.

---

## 5. Hugging Face release

Dataset repo: `karma689/tibetan-quotation-detection` (keep `karma689/layer_detection` alone —
that is the old experiment).

### 5.1 Format

Three JSONL files: `train.jsonl`, `validation.jsonl`, `test.jsonl`. One row per **book**, not
per span, so the offsets stay meaningful against the text they belong to:

```json
{
  "book_id": "IB522F095",
  "title": "...",
  "text": "<full base/v001.txt>",
  "doc_chars": 281185,
  "batch": "new",
  "source_layer": "Citation",
  "density": 5.81,
  "density_bucket": "3-8%",
  "spans": [
    {"start": 3850, "end": 3975, "cls": "MISC", "closed": false, "seen_in_train": false}
  ]
}
```

State in the card, in bold, that `end` is **inclusive**.

Also ship `stats.json` — the corpus targets, the per-split measured table, the chosen
weights, and the leakage counts. Same content as `_meta` in the split JSON. A reader should be
able to check our claims without rerunning anything.

Upload with `huggingface_hub.HfApi.upload_folder`. Read the token from the `HF_TOKEN`
environment variable. Never write a token into a file. Support `--dry-run` that builds
everything locally under `Data/hf_export/` and uploads nothing — make that the default, so
uploading takes an explicit `--push`.

### 5.2 The dataset card — this is the part that matters

`README.md` in the dataset repo, with YAML frontmatter (`license`, `language: [bo]`,
`task_categories: [token-classification]`, `configs`, `size_categories`).

Write it in plain language. Someone should understand it without reading our code. Cover:

**What the data is.** Tibetan Buddhist texts from OpenPecha with quotation spans marked by
hand. 254 books, 23,099 quotations. Give the chain honestly in one line: 28,892 raw annotation
pieces → 23,476 after merging pieces of the same quotation → 23,099 after dropping
under-annotated books. Both steps are the kind of thing a careful reader will notice and ask
about. Say the spans mark where one text quotes another — scripture,
a named teacher, or a stock phrase — and give a real example with the Tibetan shown.

**Where it came from.** Two annotation batches: 125 older books labelled with a `Quotation`
layer, 195 newer books labelled with a `Citation` layer. Say plainly that we checked these
are the same annotation under two names, and that we treat them as one label. Say that batch
is a balancing variable precisely because the two batches differ in annotation quality.

**How it was cleaned.** Three passes, all of which belong in the card:

1. *Human review.* 398 books were reviewed; 78 were rejected whole and 254 span boundaries
   were snapped. The rule in one sentence: a book was rejected when a meaningful share of its
   spans started or ended in the middle of a syllable, because at that point the boundaries
   carry no information and a clean-looking span in such a book only got lucky. Decided per
   book, not per span, on purpose.
2. *Merging.* Long quotations were stored as several consecutive pieces. We rejoined them, and
   we did not join across a closing marker. §3.0.
3. *Dropping under-annotated books.* 66 books were removed because they contain quotation
   markers at the normal rate but almost no annotations. Give the coverage table from §3.0b —
   it is the evidence, and it is the most likely thing a reviewer will challenge.

**The label set.** TEXT, PERSON, MISC, FORMULA, CHAINED — one line each on what they mean,
with counts and percentages.

**How it is split.** Book-level. The four balancing variables. The measured table from §3.5.
Say clearly that only TEXT, PERSON and MISC are balanced, that FORMULA and CHAINED run about
a point high as a result, and why we accepted that.

**Leakage.** Its own top-level section, not a footnote. Structural leakage is zero and here is
why. Quotation-text overlap is about 12% of held-out spans and here is the table. We do not
deduplicate, and here is the reason. Every held-out span carries `seen_in_train`, and anyone
reporting a number on this benchmark should report the `seen_in_train == false` subset too.

**Known limits.** Boundary noise survives in accepted books. Under-annotation is reduced, not
solved — the 1–3% density band still sits at 7.7% marker coverage, so some real quotations in
the released data carry no label. The published density of 7.41% describes adequately
annotated books, not a random Tibetan text. `closed_pct` varies widely because a closer only
appears at the end of a quotation, not on every piece it was stored in. Some books annotate
source references rather than quotation text. Point at
`Data/audit/quotation_annotation_coverage.md`, `Data/audit/quotation_leakage.md` and
`stats.json` for the raw numbers.

**How to load it,** with a short working `datasets.load_dataset` snippet.

Do not overclaim. If a number is uncertain, say it is uncertain.

---

## 6. Module shape

Public functions in `quotation_benchmark.py`:

```python
load_gold(path)                -> Gold
merge_continued_spans(gold)    -> Gold          # §3.0  — run first, everything else uses it
annotation_coverage(gold)      -> CoverageTable # §3.0b — the closer-marker diagnostic
drop_under_annotated(gold, min_density=1.0) -> Gold
check_duplicate_books(gold)    -> list[tuple[str, str]]
corpus_targets(gold)           -> Targets
sweep_and_split(gold, targets) -> Split          # §3.3 + §3.4
audit_leakage(gold, split)     -> Leakage        # §4
write_split_csv / write_split_json / write_leakage_md
build_hf_export(gold, split, leakage, out_dir)
push_to_hub(out_dir, repo_id)
main(argv)                     -> int
```

CLI flags: `--gold`, `--out`, `--repo`, `--push`, `--min-density` (default 1.0), `--seed`,
`--no-sweep`.

Standard library plus `huggingface_hub` only. No pandas, no scikit-learn, no
`iterative-stratification`. Type hints, `from __future__ import annotations`, a module
docstring holding the corpus targets and the reason for each design choice — the same way
`quote_markers.py` does it.

---

## 7. Before you call it done

1. The three splits are disjoint and cover all 254 kept books.
2. Merging turns 28,892 raw spans into 23,476 merged spans, and every merged span's text is
   the exact slice from its first piece's start to its last piece's end — no character added,
   none dropped. The density cut then leaves 254 books and 23,099 spans. Total spans across
   the splits equals that.
3. Test and val profile error under 1.5; density within 0.15 points of 7.41%.
4. Re-running with the same seed gives a byte-identical `quotation_split.json`.
5. Round-trip check: pick 20 spans at random across the export, slice
   `text[start:end+1]`, and confirm the string matches what the same offsets give against
   `Data/repos/<book>/<book>.opf/base/v001.txt`. Off-by-one here poisons everything.
6. `seen_in_train` is present on every held-out span and never true for a train span.
7. `--dry-run` writes a complete `Data/hf_export/` and touches the network zero times.
8. The old 7-layer pipeline still runs: `python -m layer_detection.split` is unaffected.

Print a short summary at the end — books and spans per split, the profile table, the leakage
line, and where the files went.
