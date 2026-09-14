# Handoff: build the quotation-only split and stratifier

You are working in `~/Developer/quotation_detection`, a Tibetan span-detection
project. An encoder (mmBERT-base) is fine-tuned as a BIO token classifier to
find annotation layers in classical Tibetan Buddhist texts.

**Read this whole brief before touching anything. A large part of it is about
what NOT to change, and about a measurement discipline that matters more than
the code.**

---

## 1. Why this exists

The current train/val/test split (`Data/audit/split.csv`, 217/25/24 books) was
built by `src/layer_detection/split_utils.py` for **all seven layers at once**.
It sizes by total labelled character volume and balances by *per-layer book
presence* — whether a book contains a layer at all, yes/no.

That was a reasonable design for a seven-layer model. It is a bad split for a
quotation-only benchmark, and the defect is measurable, not a matter of taste.

**The test set is at the 0.1st percentile of random draws** on quotation marker
composition. Three thousand random 24-book test sets give a text-citation share
of 56.1% ± 9.1; the actual test set is at 30.6%. It is more skewed than 999 out
of 1000 random draws. This was not bad luck — the multi-layer objective pushed
it there.

Your job is to build a quotation-specific split. It is **not** to retrain
anything, and **not** to touch the existing seven-layer split.

---

## 2. Facts already measured — use these, do not re-derive

All of these come from a full sweep of `Data/repos` (266 books, 24,730 raw
Quotation spans). Reference implementations are in `script/analysis/qd_*.py`
(scratch quality — read them for the method, do not import them).

### 2.1 The marker taxonomy

Tibetan quotations are introduced by case markers (རྣམ་དབྱེ་). This was
mined from the data, not guessed: candidates were the 1–3 syllables adjacent to
every gold span boundary in the **159 train books only**, scored against a
background of shad/tsheg positions ≥200 chars from any gold boundary. Val and
test were never used in derivation.

Over the **14,984 sentence-initial spans** (see §2.2 — this is the population
that matters):

| category | grammar | markers | share | median len | closed by ཞེས/ཅེས |
|---|---|---|---:|---:|---:|
| **TEXT** | འབྱུང་ཁུངས་ ablative | ལས, ནས | **56.1%** | 120 | 89% |
| **PERSON** | བྱེད་སྒྲ་ ergative / speech | ཀྱིས/གིས/གྱིས/ཡིས, name+ས, ཞལ་ནས, གསུང་ནས, ན་རེ, གསུངས/སྨྲས/བཤད/བརྗོད/ཞུས/གསོལ/ཟེར/སྩལ | **14.3%** | 121 | 65% |
| FORMULA | ལ་དོན terminative | སྐད་དུ, ཇི་སྐད་དུ, དུ/ཏུ/རུ/སུ | 4.9% | 115 | 90% |
| CHAINED | previous closer | ཞེས་དང, ཅེས་དང | 3.9% | 123 | 75% |
| **MISC** | none found | — | **20.8%** | **56** | 33% |

Validation that these are real categories and not an artifact of the rules:
the closing rate differs systematically (89% for TEXT vs 65% for PERSON) even
though the classifier only reads what comes *before* the span; and MISC has
half the median length of every marked category.

**Closers** are near-complete with three forms only: ཞེས, ཅེས, གསུངས.
ཞེས and ཅེས are **sandhi variants of one particle** — ཅེས after ད (97.6%),
ག (93.8%), བ (73.8%), ཞེས elsewhere. Collapse them. Splitting them would be
stratifying on the last letter of the quotation.

### 2.2 Sentence position splits the population in two

| | spans | % | marked |
|---|---:|---:|---:|
| starts a new sentence (after །) | 14,984 | 60.6% | **75.4%** |
| starts mid-sentence | 9,746 | 39.4% | **7.6%** |

The mid-sentence marked rate is 7.4 / 8.0 / 7.7 across train/val/test —
**identical**, therefore carries no information. All marker computation must be
done over sentence-initial spans only. Including mid-sentence spans dilutes the
signal with noise.

### 2.3 What does NOT work — do not reintroduce these

| approach | result |
|---|---|
| ཏེ, སྟེ, ནི, དང as openers | lift 0.29–0.41, **below chance** — sentence-final particles, not quotative |
| reading back over connectives | lift 0.37 for the added spans — noise |
| marker at the end of the previous sentence | lift 0.78 / 0.22 / 0.18 — below chance |
| bare ལ / ན / ར la-don | lift 0.40–0.97 — only the **-u** forms (དུ/ཏུ/རུ/སུ) are real, at 3.55 |
| publisher as a stratification variable | explains 15.2% of PERSON share, 6.6% of density; within-publisher sd 23.4 vs corpus 28.8 — **publishers are not uniform, drop it** |
| book-count quotas | cuts spread only ~25%; span-volume matching cuts it 92% |

### 2.4 Corpus targets

```
TEXT 56.1%   PERSON 14.3%   FORMULA 4.9%   CHAINED 3.9%   MISC 20.8%
quotation density 5.16%
266 books · 12 publishers · 29 series · 72 zero-quotation books (27%)
```

Current test set for comparison: TEXT 30.6 · PERSON 17.8 · MISC 46.3 ·
density 3.01% · 6 publishers · 12 series.

---

## 3. Known data defect — read before trusting any number

**36% of all quotation spans have an edge that cuts mid-syllable** (the
character just outside the span is a letter, not a tsheg/shad/space).

| split | spans | misaligned |
|---|---:|---:|
| train | 19,716 | 33.1% |
| val | 2,984 | 44.9% |
| test | 2,030 | **51.0%** |

It is a book-level property, not random noise: P000101 is 92.2% misaligned over
1,241 spans, while P000022 is 0.0% over 366. And it is not a systematic
off-by-one — the distance to the nearest delimiter is spread across 1 char
(2,326), 2 (1,984), 3 (1,543), 4+ (694).

71.7% of MISC spans have a ragged edge, against 0.6% of fully-marked spans. So
part of the MISC class is repair debt, not a citation style.

**This does not block the split**, but it must be recorded in the output, and
a boundary-repair pass is the higher-priority piece of work. Do not silently
"fix" offsets as part of this task.

---

## 4. Environment

```
~/Developer/quotation_detection
venv/  → homebrew python 3.14; pip install -e .
Data/  → 8.3 GB, do not move or copy
```

Existing work from the analysis session, already in the repo:

| path | status |
|---|---|
| `src/layer_detection/quote_markers.py` | validated closer/opener regex profiles + derivation notes in the docstring |
| `script/build_quotation_tags.py` | per-span and per-book tag tables |
| `Data/audit/quotation_tags/*.csv` | generated outputs incl. `book_top3_profile.csv` |
| `script/analysis/qd_*.py` | scratch reference implementations — read, do not import |

---

## 5. What to build, in this order

### A. `src/layer_detection/quotation/` — a proper package

```
src/layer_detection/quotation/
    __init__.py
    markers.py     ← move quote_markers.py here, keep the docstring
    profile.py     ← per-book marker profile
    stratify.py    ← the split algorithm
    report.py      ← comparison tables
```

`script/quotation_profile.py` and `script/quotation_split.py` are thin
wrappers, matching the existing `script/split_books.py` convention.

### B. `profile.py` — score every book

For each of the 266 books, over its **sentence-initial spans only**:

```
book, n_spans_total, n_spans_new_sentence,
pct_text, pct_person, pct_formula, pct_chained, pct_misc,
density,               # quot_chars / doc_len
misaligned_rate,       # §3
publisher, series, author, category   # from meta.yml, for reporting only
```

Parsing rules that matter:

- Delimiters are `་ ༌ ། ༎ ༏ ༑ ༈ ༔ ༴ ༵ ༶ ྅` plus whitespace. **༔ (terma shad)
  must be included** — omitting it loses `ལས༔` in terma texts.
- Walking back from a span start, skip whitespace, shad **and trailing tsheg**.
  Omitting tsheg returns an empty syllable for `ཀྱང༌།` patterns.
- "New sentence" = a shad was crossed while walking back.
- Read over `ཀྱང` / `ཡང` to the syllable before.
- `དང` preceded by `ཞེས`/`ཅེས` is CHAINED, not a marker.
- Series comes from `web_metadata.collection.en` with the trailing `– Book N`
  stripped; five series appear under both Tibetan and English names and must be
  merged (see `script/analysis/qd_series.py`).
- Books with <10 sentence-initial spans get profiles but are flagged
  `reliable=0` — their rates are noise.

Write to `Data/audit/quotation_tags/book_profile.csv`.

### C. `stratify.py` — the split

**Three layers. The unit of assignment is always a whole book.**

1. **Density buckets are hard strata.** Six buckets: zero-quotation, 0–1%,
   1–3%, 3–8%, 8–15%, >15%. Each bucket gets its own 80/10/10 span quota.
   Books never move between buckets. Density is only 6.6% explained by
   publisher and uncorrelated with the marker profile (r = −0.06), so it must
   be controlled directly or the optimiser will wreck it.

2. **Inside each bucket, assign whole books greedily by span volume**, each
   step choosing the book that minimises

   ```
   |pooled_text   − 56.1|
   + |pooled_person − 14.3|
   + |pooled_misc   − 20.8|
   ```

   where `pooled_x(S) = Σ(pct_x[b] × n_spans[b]) / Σ(n_spans[b])`.

   **Span-weighted, not book-weighted.** This is the single most important
   detail — book-count quotas cut the variance by 25%, span-volume matching
   cuts it by 92%.

   Three targets, not one. Optimising TEXT alone drives MISC to 14.2% against a
   20.8% target, because the five categories sum to 100% and one target leaves
   44% of the mass free to drift.

   **No publisher term.** See §2.3.

3. **Zero-quotation books** assigned proportionally by count (~27%). Do not
   minimise them — they are the only measurement of false positives on true
   negatives, and the current split already has them at 25%, which is correct.

Deterministic: no RNG in the final path. Seed only if you expose a
`--n-restarts` option for diagnostics.

Write `Data/audit/quotation_split.csv` — one row per book with the assignment
and every profile column. **Do not modify `Data/audit/split.csv`.**

### D. `report.py` — prove it worked

Emit a markdown table comparing, on the marker profile, density, publisher
count, series count and zero-quotation share:

- the corpus
- the existing seven-layer split's test set
- 3,000 random 24-book test sets (mean and sd)
- the new split's test and val

Pre-registered expectations, from the analysis runs:

| metric | current | target |
|---|---:|---:|
| profile error (\|text\|+\|person\|+\|misc\| deviation) | 54.5 | **< 4** |
| TEXT share | 30.6% | 54–58% |
| MISC share | 46.3% | 19–23% |
| density | 3.01% | 4.5–5.5% |
| publishers represented | 6 | ≥ 9 |
| series represented | 12 | ≥ 20 |

If profile error lands above 6, something is wrong with the span weighting —
check that you are weighting by `n_spans_new_sentence` and not book count.

---

## 6. Hard constraints

1. **Whole books only.** A book goes to exactly one split. The marker profile
   is computed from a book's own spans and used only to decide where that book
   goes. Never split a book, never move spans independently.
2. **Do not touch** `split_utils.py`, `split.py`, `Data/audit/split.csv`, or
   any `experiments/` folder. The seven-layer split must keep working
   unchanged.
3. **Do not repair boundary offsets** in this task. Record `misaligned_rate`
   and stop.
4. **Do not add publisher, series, genre or author to the objective.** They are
   reporting columns. §2.3 has the evidence.
5. **Generate the split once and freeze it.** The rule is specified above;
   implement it, run it, commit the output. Do **not** iterate on the
   stratification because you dislike a resulting score — that is selecting on
   the test set, and v1–v2 results have already been seen on the old split.
6. No new dependencies. stdlib + what `pyproject.toml` already pins.

---

## 7. The comparability protocol — do not skip this

The new test set will contain more marked quotations (56% TEXT vs 30%), which
are the easier cases. **Scores will probably go up for reasons that have
nothing to do with the model.** From outside this looks exactly like
goalpost-moving, so the decomposition has to be explicit:

1. Freeze the new split before training anything.
2. Score the **existing best checkpoint on both** the old and new test sets.
   That single number is how much of any future gain is the split.
3. Record both in `docs/RESULTS.md` alongside the reason for the change.
4. Keep DharmaBench QUDT (`third_party/DharmaBench`, `script/eval_dharmabench_qudt.py`)
   as the external anchor. If the new split shows +8 F1 and QUDT is flat, the
   gain is not real. Use QUDT for direction, not level — its 60.09 chance
   baseline and 1,694-char documents don't transfer.

---

## 8. What to hand back

- The four modules and two wrappers, with tests (follow
  `tests/test_single_layer.py` for style — unit tests on the classifier, on
  pooled-share arithmetic, and a leakage assertion that no book id appears in
  two splits).
- `Data/audit/quotation_tags/book_profile.csv`
- `Data/audit/quotation_split.csv`
- `docs/quotation-split-report.md` — the §5D table, plus the misalignment rate
  per split, plus the old-checkpoint-on-both-splits number from §7.
- A short note on anything in §2 you could not reproduce. If a number here
  disagrees with what you measure, **stop and report rather than adjusting the
  spec** — several of these were re-derived after correcting parser bugs, and a
  disagreement most likely means a delimiter or tsheg-handling difference worth
  finding.

---

## 9. AMENDMENT — new data batch (read this before §2)

A second batch of books was added after the analysis in §2 was run. **It changes
the inputs materially. §2's numbers describe the old batch only.**

### 9.1 The corpus is still syncing

`Data/repos` went 266 → 328 → 337 → **422** directories over the course of one
session. **Count the books and re-run the profile before trusting any total.**
Do not hard-code corpus targets from §2.4 — recompute them.

### 9.2 The new books have no `Quotation` layer — they have `Citation`

`audit_utils.EXCLUDED_LAYERS = ("Citation",)` must become **per-batch**. In the
new books, `Citation` is the same annotation the old books call `Quotation`.
Evidence:

| | median len | closed by ཞེས/ཅེས | TEXT | PERSON | FORMULA | CHAINED |
|---|---:|---:|---:|---:|---:|---:|
| old `Quotation` | 116 | 73% | 56.2% | 14.2% | 5.0% | 3.9% |
| new `Citation` | 117 | **74%** | 40.8% | 13.3% | 5.2% | 4.2% |
| old `Citation` (31 spans, 2 books) | 210 | 41% | 51.9% | 29.6% | 11.1% | 0.0% |

The old `Citation` layer is a different, tiny thing — keep excluding it for
`P0*` books. Map new-batch `Citation` → Quotation.

### 9.3 The new batch is much cleaner — and that is a confound

| batch | books | spans | sentence-initial | misaligned |
|---|---:|---:|---:|---:|
| old `P0*` Quotation | 194 | 24,730 | **65%** | **36%** |
| new `I*` Citation | 114 | 11,585 | **100%** | **2%** |
| combined | 308 | 36,315 | — | — |

New data adds 47% more spans and 59% more books.

This is strong independent support for the §3 reading that the old batch's
mid-sentence unmarked spans are largely annotation damage: a cleanly annotated
batch has almost none of them.

**Consequence: batch is now a required balance variable.** It correlates with
boundary quality (36% vs 2%) and structure (65% vs 100% sentence-initial). If
test ends up mostly-new and train mostly-old, you measure annotation batch
rather than model skill. Add `batch` (old/new) to the greedy objective, or
stratify within batch. Report the per-split batch mix either way.

### 9.4 Metadata differences in the new batch

New books use a different `meta.yml` schema — `source_metadata` with `author`,
`authors`, `series`, `sku` directly; **no `web_metadata` block**, so no
publisher, collection or category.

- **Drop `series` entirely.** The field exists in the new batch but is not in a
  consistent order, and the old batch's series had to be reconstructed from
  `collection` strings. It is not a reporting column worth keeping.
- **Strip `མཛད་པ་པོ།` from author values.** New-batch authors read
  `མཛད་པ་པོ། ཀརྨ་རང་བྱུང་ཀུན་ཁྱབ།` — the prefix is the label "author", not part
  of the name. Also split multi-author fields on `,` — several books list five.
- **Publisher is unavailable for the new batch.** It was already dropped from
  the objective (§2.3); now it cannot even be a reporting column for half the
  corpus. Leave it out.
- New layers appear that the old batch lacks: `Commentary` (114 books),
  `Footnote`, `BookNumber`, `SubTitle`. Out of scope here, but the seven-layer
  pipeline will need a decision about them.

### 9.5 MISC → PERSON rule, validated

Mined from MISC spans across both batches. **Speech verb + nominalizer** moves
to PERSON:

```
speech ∈ {གསུངས, སྩལ, སྨྲས, བཤད, བརྗོད, ཞུས, གསོལ, ཟེར}
nomz   ∈ {པ, པར, བ, བར, པོ, པའི, བའི}
if prev_syllable ∈ nomz and syllable_before_it ∈ speech → PERSON
also: ན་རེ → PERSON
```

| pattern | spans | lift |
|---|---:|---:|
| སྨྲས་པ | 72 | 26.1 |
| ན་རེ | 31 | 5.8 |
| སྩལ་པ (བཀའ་སྩལ་པ) | 26 | 18.5 |
| གསུངས་པ | 18 | — |
| གསོལ་པ | 16 | — |
| བཤད་པ / བརྗོད་པ | 14 | — |

Moves ~180 spans, PERSON 13.5% → 14.3%. Small but correct.

**Do NOT add speech + སོ or speech + ཏེ.** `གསུངས་སོ` scores lift 0.9 (chance)
— it is the *closer of the previous quotation*, not an opener.

**REJECTED — locative-on-title → TEXT.** `རྒྱུད་བླར` ("in the Uttaratantra")
is genuine, but a `len(prev) > 2` heuristic also catches རྣམས་ལ, དོན་ལ, ཉིད་ལ,
ཆེན་ལ — ordinary dative uses. Without a title lexicon this cannot be separated.
Leave those in MISC.
