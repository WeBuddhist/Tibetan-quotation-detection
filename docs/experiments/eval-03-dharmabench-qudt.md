# eval-03 — DharmaBench QUDT (external benchmark)

**Date:** 2026-08-20  **Status:** complete, **interpretation blocked on one missing baseline**
**No training.** v1.2 and v1.3 checkpoints scored on DharmaBench's Tibetan Quotation
Detection test split (406 documents, 335 gold quotations, mean length ~1,000 characters).

## Question

Is the model weak, or is our corpus lying about it? DharmaBench is external, independently
annotated, and comes with published baselines — the one measurement that can separate the two.

## Results

MUC-5 `type` mode, the metric DharmaBench uses:

| | argmax | best Viterbi |
|---|---:|---:|
| v1.2 | 0.608 | 0.611 (penalty 0) |
| v1.3 | 0.550 | **0.653** (penalty 4) |

Published baselines on the same task: Gemini 2.0 Flash 0.762 · Gemini 2.5 Flash 0.618 ·
Claude 3.7 Sonnet 0.576 · Gemini 2.5 Pro 0.572 · GPT-4o 0.529. All generative.

IoU@0.5 over the full test set:

| | argmax | best Viterbi |
|---|---:|---:|
| v1.2 | 0.000 | 0.023 |
| v1.3 | 0.038 | **0.159** |

v1.3 Viterbi penalty sweep, MUC-type: 0 → 0.631 · 1 → 0.646 · 2 → 0.647 · **4 → 0.653** ·
8 → 0.650 · 12 → 0.651. IoU@0.5 peaks earlier: 0 → 0.168 · **1 → 0.181** · 2 → 0.166 ·
4 → 0.159 · 8 → 0.148 · 12 → 0.139.

## The caveat that reframes the headline

Read directly from `predictions.jsonl`, independent of any metric:

- **244 of 406 documents (60.1%) contain no gold quotation.**
- Viterbi is silent on 231 of those 244 — correct, and free.
- Viterbi is also silent on **115 of the 162 documents that do contain quotations** (71%).
- It emits **80 spans against 335 gold — 0.24× under-proposal.** On our own validation the
  same decoder sits at 1.0×.

So a substantial share of 0.653 is credit for staying quiet on the empty majority.

**Not established:** exactly how much. An attempt to reproduce the reported MUC-type number by
macro-averaging per-document P/R with empty+empty = 1.0 gives 0.799 against the harness's
0.653, so the harness aggregates differently. **Action: run a null baseline — predict nothing
everywhere — through the same scoring code.** That number is the floor, and every result here
should be reported as distance above it.

## What is unambiguous

Restricted to the 162 documents that actually contain a quotation, IoU@0.5:

| | tp | gold | pred | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| argmax | 37 | 335 | 1,161 | 0.032 | 0.110 | 0.050 |
| Viterbi | 33 | 335 | 65 | **0.508** | **0.099** | 0.165 |

## Findings

- **Precision 0.508 on external, independently annotated data is the real result.** When the
  model commits to a span, it is right about half the time at IoU ≥ 0.5 against annotations
  that are not ours. That is evidence it learned something transferable about quotations, not
  only our corpus's annotation habits.
- **Recall 0.099 is the problem**, and it is confounded. Two candidate explanations and this
  run cannot separate them:
  1. *Hedging* — the mechanism proposed in eval-02, where contradictory supervision teaches
     the model to stay quiet on unseen material.
  2. *Length shift* — DharmaBench documents average ~937 characters; our books run ~500,000
     and training used 8,192-token windows. The model has never seen an input this short.
- **Viterbi helps here too**, and by more than on our own data: v1.3 argmax → Viterbi moves
  MUC-type 0.550 → 0.653 and IoU 0.038 → 0.159. The argmax fragmentation is not an artifact
  of our corpus.
- **v1.3 clearly beats v1.2 externally** (IoU 0.159 vs 0.023), consistent with the in-house
  finding, and this comparison is not affected by our answer-key damage.
- **The optimal Viterbi penalty differs by metric**: 4 under MUC-type, 1 under IoU@0.5. Worth
  noting when tuning — the decode setting is metric-dependent.

## What this changed about the plan

- Does **not** cleanly resolve "weak model vs lying corpus" — the length confound blocks it.
- The string-overlap test (eval-05 area) becomes more valuable, not less: it isolates
  memorisation *within* our own data with no length shift.
- Adds a required baseline before any external claim: null-model floor through the same harness.
- If the null floor turns out low and 0.653 is largely earned, this is publishable — it fills
  the encoder-shaped gap DharmaBench's authors explicitly left open.

## Artifacts

`experiments/mmbert_base_v1.2/dharmabench_qudt/{metrics.json,predictions.jsonl}`
`experiments/mmbert_base_v1.3/dharmabench_qudt/{metrics.json,predictions.jsonl}`

## Caveats

Quotation only — one of seven layers. 406 documents, 335 gold spans, no confidence intervals.
Length distribution differs sharply from our training data. Annotation conventions are
DharmaBench's, not tsadra's, and no manual comparison of the two has been done.
