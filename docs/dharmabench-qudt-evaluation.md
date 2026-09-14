# DharmaBench QUDT — Evaluation Report

External evaluation of `mmbert_base_v1.2` and `mmbert_base_v1.3` on the DharmaBench
Tibetan Quotation Detection task (QUDT), under the paper's own protocol.

**Status:** this report supersedes the initial QUDT writeup, which omitted the
benchmark's chance baseline and consequently misread the headline result.

---

## 1. Summary

| | |
|---|---|
| Best result | **v1.3 + Viterbi (p=4): 65.34 MUC type F1** |
| Benchmark chance baseline | **60.09** (predict nothing on every document) |
| Margin over baseline | **+5.25** |
| Rank on the published board | **2nd of 12**, behind Gemini 2.0 Flash (76.23) |
| Character-level recall | **0.070** |
| Gold quotations located at all | **65 of 335 (19%)** |
| Setting | **zero-shot** — no QUDT training example was seen |

Two readings are both true and must be held together:

1. A ~300M-parameter encoder running on a laptop places second on this benchmark
   against three-seed frontier-LLM means, having never seen a QUDT training
   example. That is a real result.
2. **87% of the headline score comes from the model correctly emitting nothing on
   the 60% of test documents that contain no quotation.** On the actual detection
   task the model is weak, and the evaluation says clearly why.

---

## 2. Setup

**Data.** `third_party/DharmaBench/data/Tibetan/QUDT/test.jsonl`, vendored from
the official repository. Verified contents: **406 documents, 335 gold QUOTE
spans, 244 documents (60.1%) with no QUOTE at all.**

**Protocol match.** Following the paper: MUC-5 via `eval4ner`, mode `type`
(partial overlap with matching tag counts as a match), **QUOTE label only** — the
other four QUDT labels (OP, CP, TITLE, GEN_SRC) are filtered from gold. Character
IoU@0.5 F1 is reported alongside as our own boundary-magnitude metric.

**Inference.** `script/eval_dharmabench_qudt.py`. Single forward pass per
document, `truncation=True, max_length=8192`, `sdpa` attention. The 15-label BIO
head is masked to indices `[0, 1, 2]` = `O`, `B-QUOTE`, `I-QUOTE`, so only those
three labels can be decoded. Token predictions are mapped back to character
offsets via the tokenizer's offset mapping.

**No windowing was used or needed.** QUDT documents max out at 1,694 characters,
comfortably inside one 8,192-token context. The sliding-window machinery built
for our own book-length inputs is bypassed entirely here.

**Hardware.** No GPU required. All 406 documents complete in under 4 minutes at
batch-1 on CPU or Apple Silicon MPS.

**Decoders.** Raw argmax plus Viterbi under BIO transition constraints at exit
penalties {0, 1, 2, 4, 8, 12}.

---

## 3. Results

### 3.1 The number that reframes everything

DharmaBench Table 5 reports a chance baseline of **60.09**. That is exactly
244/406 — the fraction of empty test documents. It is the score for emitting an
empty prediction everywhere, because MUC is averaged per document and empty-gold
against empty-prediction scores a full 1.0.

Every result below must be read against 60.09, not against 0.

### 3.2 v1.3 — full sweep

| decoder | MUC type F1 | vs 60.09 | MUC exact F1 | IoU F1 | IoU P | IoU R | matched | spans predicted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| argmax | 0.5501 | −5.08 | 0.5074 | 0.0376 | 0.023 | 0.110 | 37 | 1,635 |
| viterbi p=0 | 0.6311 | +3.02 | 0.5524 | 0.1683 | 0.234 | 0.131 | 44 | 188 |
| **viterbi p=1** | 0.6458 | +4.49 | 0.5624 | **0.1806** | 0.323 | 0.125 | 42 | 130 |
| **viterbi p=4** | **0.6534** | **+5.25** | 0.5700 | 0.1590 | 0.412 | 0.099 | 33 | 80 |
| viterbi p=2 | 0.6472 | +4.63 | 0.5675 | 0.1655 | 0.360 | 0.107 | 36 | 100 |
| viterbi p=8 | 0.6503 | +4.94 | 0.5724 | 0.1478 | 0.423 | 0.090 | 30 | 71 |
| viterbi p=12 | 0.6510 | +5.01 | 0.5714 | 0.1390 | 0.412 | 0.084 | 28 | 68 |

Gold contains 335 spans. At p=4 the model emits **80**.

### 3.3 v1.2 — full sweep

| decoder | MUC type F1 | vs 60.09 | MUC exact F1 | IoU F1 | matched | spans predicted |
|---|---:|---:|---:|---:|---:|---:|
| argmax | 0.6076 | +0.67 | 0.5813 | **0.0000** | **0** | 218 |
| viterbi p=0 | 0.6110 | +1.01 | 0.5862 | 0.0100 | 2 | 66 |
| viterbi p=1 | 0.6058 | +0.49 | 0.5911 | 0.0108 | 2 | 35 |
| viterbi p=2 | 0.6056 | +0.47 | 0.5961 | 0.0169 | 3 | 21 |
| viterbi p=4 | 0.6030 | +0.21 | 0.5985 | 0.0233 | 4 | 8 |
| viterbi p=8 | 0.6022 | +0.13 | 0.5985 | 0.0177 | 3 | 4 |
| viterbi p=12 | 0.6022 | +0.13 | 0.5985 | 0.0177 | 3 | 4 |

**v1.2 is functionally mute on QUDT.** It never clears the baseline by more than
one point, locates at most **4 of 335** quotations in any configuration, and at
p≥4 emits fewer than ten spans across the entire test set. Its 0.58–0.60 exact
F1 is silence, not accuracy. The earlier report's v1.2-vs-v1.3 comparison also
compared v1.2 at p=0 against v1.3 at p=4; the full grid above removes that
confound and the conclusion strengthens rather than weakens.

### 3.4 Where the 65.34 comes from

v1.3, Viterbi p=4, decomposed by document type:

| | count |
|---|---:|
| empty documents correctly left empty | **231 / 244** |
| empty documents false-fired on | 13 |
| gold quotations with any overlapping prediction | **65 / 335** |
| gold quotations completely missed (zero overlap) | **270 / 335 (81%)** |

0.6534 × 406 = 265.3 document-points, of which the 231 silent documents supply
231. The remaining ~34 points spread across 162 quote-bearing documents give a
mean per-document score of **≈ 0.21** on the task the benchmark is named for.

### 3.5 Position against the published board

Verified against the paper's Table 5. The baseline row is included, as it must be:

| model | QUDT type F1 | seeds |
|---|---:|---|
| Gemini 2.0 Flash | 76.23 ±0.47 | 3 |
| **mmBERT v1.3 + Viterbi (ours, zero-shot)** | **65.34** | 1 |
| Gemini 2.5 Flash | 61.81 ±0.93 | 3 |
| *baseline — predict nothing* | *60.09* | — |
| Llama 4 Scout | 59.61 ±0.25 | 3 |
| Claude 3.7 Sonnet | 57.63 | 1 |
| Qwen2.5-72B | 57.32 ±1.00 | 3 |
| Gemini 2.5 Pro | 57.22 | 1 |
| GPT-4o mini | 55.82 ±0.49 | 3 |
| Claude 4 Sonnet | 55.24 | 1 |
| GPT-4o | 52.86 | 1 |
| DeepSeek-R1 | 49.30 | 1 |
| Claude 3 Haiku | 22.16 ±1.01 | — |

The defensible claim: **second overall, and one of only three systems on the board
that clears the do-nothing baseline.** Beating GPT-4o, Claude, Llama and Qwen is
true but carries little weight — all of them lose to an empty string.

Caveat: our figure is a single point estimate against three-seed means, and
carries no confidence interval. Their seed spread is ±0.25–1.00; a bootstrap over
406 documents would plausibly put ours at ±3–4, which is the size of the gap to
Gemini 2.5 Flash.

---

## 4. Error analysis

### 4.1 The model under-marks by 8×

| | argmax | viterbi p=4 |
|---|---:|---:|
| gold QUOTE characters | 139,665 | 139,665 |
| predicted characters | 17,122 | 16,979 |
| overlapping characters | 9,775 | 9,734 |
| **character recall** | **0.070** | **0.070** |
| character precision | 0.571 | 0.573 |
| share of corpus marked QUOTE | 4.1% | 4.1% |
| share of corpus gold marks QUOTE | **33.7%** | **33.7%** |

The model is not mislocating — 57% of what it marks is genuinely inside a gold
quotation. It simply marks almost nothing. **Character recall is identical under
both decoders.** No decoding change touches it.

### 4.2 Viterbi consolidated; it did not silence

Span *counts* fall from 1,635 to 80, but characters marked fall only from 17,122
to 16,979 — **0.8%**. Viterbi reorganised the same marked text: it deleted
isolated weak fragments (cleaning 25 empty documents, 206 → 231) and bridged gaps
elsewhere, netting flat. The MUC gain of +42 document-points splits roughly
25 from fragment cleanup on empty documents and 17 from producing whole spans on
documents it gets right. Detection went *down*: argmax touches 95 gold
quotations, p=4 touches 65.

### 4.3 Span lengths — the primary failure axis

| | median | mean | p90 | max |
|---|---:|---:|---:|---:|
| gold | 278 | 417 | 1,054 | 1,608 |
| model, argmax | **2** | 10 | 16 | 633 |
| model, viterbi p=4 | 176 | 212 | 466 | 820 |

Under argmax the median predicted span is **two characters**. The model's
quotation signal crosses threshold only in isolated flickers and never sustains.
Viterbi glues nearby flickers into 176-character spans — still well short of the
278-character gold median — and never produces anything past 820 characters while
gold runs to 1,608.

Recall by how much of the passage the gold quotation occupies (v1.3, p=4):

| gold quote size | n | char recall | quotes touched |
|---|---:|---:|---|
| < 20% of passage | 146 | **0.275** | 41/146 |
| 20–50% | 96 | 0.076 | 14/96 |
| 50–90% | 62 | 0.034 | 9/62 |
| > 90% of passage | 31 | **0.013** | 1/31 |

**21× better on short quotations than on passage-length ones.** This is the
dominant effect.

Context: our training corpus averages **134 characters** per QUOTE span
([DIAGNOSIS_v2.md](DIAGNOSIS_v2.md) §1). DharmaBench averages **417**. We evaluated a model whose
learned notion of a quotation is roughly one-third the length of the target.

### 4.4 Window truncation

| | n | char recall |
|---|---:|---:|
| quotation sits fully inside the passage | 163 | **0.146** |
| quotation runs off the window edge | 172 | **0.027** |

**5.4× better when both edges are visible**, and 172 of 335 test quotations
(51%) touch an edge.

### 4.5 Citation frames are a secondary effect

QUDT annotates opening and closing particles as separate OP/CP labels outside the
quotation span. Recall crossed by frame presence and position:

| frame | position | n | mean length | char recall |
|---|---|---:|---:|---:|
| OP + CP | contained | 69 | 405 | 0.159 |
| OP only | contained | 32 | 411 | 0.082 |
| CP only | contained | 26 | 170 | **0.273** |
| neither | contained | 36 | 116 | 0.131 |
| CP only | edge-cut | 81 | 532 | **0.007** |
| OP only | edge-cut | 66 | 523 | 0.054 |
| neither | edge-cut | 25 | 496 | 0.025 |

At matched length (405 vs 411 characters), the closing particle roughly doubles
recall — 0.159 vs 0.082 — so the frame does contribute. But the best-performing
bucket has *no* opening particle and the worst has a closing one; 61 of 335
quotations carry no frame at all; and every edge-cut bucket sits at or below
0.054 regardless of framing. **Length and truncation dominate; the frame is
real but secondary.**

### 4.6 Boundaries — near-misses are systematic

For the 65 quotations the model touches at all, IoU of the best-overlapping
prediction: median **0.518**, mean 0.571. Distribution: ≥0.90 → 16 · ≥0.75 → 26 ·
≥0.50 → 33 · ≥0.25 → 50. When it finds a quotation, it often frames it well.

But exact boundaries almost never land:

| delta (pred − gold) | mode | exactly 0 |
|---|---|---:|
| start | **−1** (23 of 65) | 4 |
| end | **+1 to +3** (26 of 65) | 5 |

A systematic one-to-three-character overhang at both ends, consistent with our
spans swallowing the shad/tsheg or the OP/CP particle. A deterministic trim
should convert a large share of near-misses into exact matches at no training
cost — **but see §5.2 first.**

---

## 5. Benchmark-side observations

### 5.1 Quotations are cut across separate test samples

Counting quotations touching a chunk edge across all 1,006 QUDT chunks:

| source | ending at chunk's last char | starting at char 0 |
|---|---:|---:|
| D3934 mDo kun las btus pa | 74 | 78 |
| D3940 bSlab pa kun las btus pa | 70 | 71 |
| D3961 mDo kun las btus pa chen po | 45 | 45 |
| all others | 17 | 8 |
| **total** | **206** | **202** |

Every cut through a quotation produces exactly one dangling end and exactly one
dangling start. The near-perfect per-source pairing is the signature of roughly
200 quotations sliced in half by the chunker and distributed as separate samples,
each annotated as a complete QUOTE.

The halves scatter across the split: 91 dangling ends in test and 115 in train;
81 dangling starts in test and 121 in train. A single original quotation
frequently has its first half in train and its second half in test.

Example: test `id=7`, 1,664 characters, with a quotation starting at character
754 and running to the final character. It carries an OP but no closing frame —
that sits in the next chunk, which is a different sample.

This is an unavoidable consequence of building a benchmark by cutting anthologies
(texts that are mostly quotation) into ~1,000-character windows, but it goes
unreported, and it means part of QUDT is "guess that this fragment continues
something you cannot see."

### 5.2 Gold offsets are internally inconsistent

Slicing `text[start:end]` reproduces the annotated string for only **281 of 335**
gold QUOTE spans. 28 require `end+1` (inclusive semantics) and **26 match
neither**. Roughly 16% of gold offsets are unreliable — normalise them before
drawing conclusions from the boundary deltas in §4.6.

### 5.3 Other scoring caveats

- 60% of documents are empty and empty-vs-empty scores a full 1.0.
- Malformed LLM output is replaced with an **empty prediction** — on this test
  set, failing to emit valid JSON is rewarded.
- `type` mode counts any overlap, one character included, as a full match.
- Train and test are a random chunk-level split of the **same 14 works** — no
  work-level held-out set, so QUDT measures in-distribution performance.
- Several headline rows are single-seed.

To the authors' credit, the 60.09 baseline is published, which most benchmark
papers omit. QUDT remains a valuable first benchmark in a badly under-resourced
area; the leaderboard position is simply a weaker signal than the raw number
suggests, in both directions.

---

## 6. Harness issues to fix

- **`eval_dharmabench_qudt.py:371`** — the best Viterbi penalty is selected on
  the test set, by MUC type F1, a metric that rewards silence. By IoU the optimum
  is **p=1** (0.1806), not p=4 (0.1590); p=4 locates nine fewer real quotations
  and still scores higher. Report both optima, or select on IoU.
- **`eval_dharmabench_qudt.py:176`** — the tokenizer loads from
  `cfg["checkpoint"]` while the model loads from the resolved `checkpoint`
  variable. The log line above it claims both came from `checkpoint`. Use one
  source.
- **`allowed_indices = [0, 1, 2]`** is correct — `LABELS = ["O"] + [...]` with
  `Quotation → QUOTE` first in `LAYER_TO_TAG` — but it is silently coupled to
  dict ordering. Add `assert LABELS[:3] == ["O", "B-QUOTE", "I-QUOTE"]`.
- **Only `argmax` and `best_viterbi`** reach the printed table and
  `predictions.jsonl`, though the full sweep is written to `metrics.json`.
- **Missing from every run:** the empty/non-empty document split, bootstrap CIs
  resampled over documents, and stratification by span length. All three are
  short additions and any one of them would have caught the original misreading.
- **Not evaluated:** the per-layer logit bias, which is our best-known internal
  configuration (v1.3 internal IoU 0.398 → 0.427). QUDT was run with Viterbi
  alone.
- **Ablation worth running:** decoding over all 15 labels and filtering to QUOTE
  afterwards, rather than masking 12 labels before decoding. Masking forces
  tsawa-looking text into the O-vs-QUOTE contest, and Tsawa→Quote is already the
  largest cross-layer confusion at 16.5%.

---

## 7. Conclusions

**On the result.** Second place on QUDT, zero-shot, from a laptop-scale encoder,
is worth reporting — as second place and as one of three systems above the null
baseline, with the baseline row present and a confidence interval attached. The
claims to withdraw from the original writeup are "decisively outperforms GPT-4o,
Claude 3.7 Sonnet, Claude 4, GPT-4o mini" stated without the baseline, and
"confirms the diagnosis that this is a decoding and evaluation problem rather
than a fundamental deficiency in the model." Clearing do-nothing by five points
does not establish that.

**On the diagnosis.** Viterbi decoding was a real fix and a real gain — the
argmax output, at a median of two characters per span, was unusable as a digital
edition regardless of how it scored. But the decoder lever is now exhausted:
character recall is 0.070 at every one of seven settings.

The limiting factor is **recall on long spans**. The model fires at quotation
onsets and cannot hold the state "still inside a quotation" across a long
interior. QUDT shows this from outside; our own validation shows the same
gradient from inside ([RESULTS.md](RESULTS.md) §2 crossed with [DIAGNOSIS_v2.md](DIAGNOSIS_v2.md) §1): token
recall of 1.000 on Author at 26 tokens, 0.951 on Chapter at 38, falling to 0.609
on Quotation at 91 tokens and 0.546 on Tsawa at 126.

**On what comes next.** [DIAGNOSIS_v2.md](DIAGNOSIS_v2.md) §6 set a decision rule in advance —
if per-token accuracy inside spans came back near 0.80, "the model really is
weak." [RESULTS.md](RESULTS.md) §1 measured **0.58**. That test has been answered and not
yet acted on. The next step is the train-split memorisation check
([RESULTS.md](RESULTS.md) §6, never run): one inference pass, no training, separating "the
model cannot fit long spans" from "the labels taught it not to." Everything
downstream — annotation repair versus architectural change — forks on that
number.

Full reasoning: `results/next-step-decision.md` in the project.

---

## Sources

- Hashiloni et al., *DharmaBench: Evaluating Language Models on Buddhist Texts in
  Sanskrit and Tibetan*, IJCNLP-AACL 2025 —
  [aclanthology.org/2025.ijcnlp-long.114](https://aclanthology.org/2025.ijcnlp-long.114/)
  (Table 5)
- [github.com/Intellexus-DSI/DharmaBench](https://github.com/Intellexus-DSI/DharmaBench)
- Internal: [DIAGNOSIS_v2.md](DIAGNOSIS_v2.md), [RESULTS.md](RESULTS.md), [FLAGGED_BOOKS.md](FLAGGED_BOOKS.md),
  `experiments/mmbert_base_v1.{2,3}/dharmabench_qudt/{metrics.json,predictions.jsonl}`
