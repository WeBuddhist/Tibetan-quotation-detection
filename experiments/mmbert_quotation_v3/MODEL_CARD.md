---
language:
- bo
library_name: transformers
pipeline_tag: token-classification
tags:
- tibetan
- quotation-detection
- token-classification
- bioe
base_model: jhu-clsp/mmBERT-base
datasets:
- karma689/tibetan-quotation-detection
---

# Tibetan quotation detection (mmBERT, BIOE)

[`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) fine-tuned to mark quotation spans in classical Tibetan. Data: [`karma689/tibetan-quotation-detection`](https://huggingface.co/datasets/karma689/tibetan-quotation-detection) (`windowed_w8192_s4579_bioe` @ `w2.0`).

## Labels

| ID | Label |
|---:|---|
| 0 | `O` |
| 1 | `B-QUOTE` |
| 2 | `I-QUOTE` |
| 3 | `E-QUOTE` |

## Training config

| | |
|---|---|
| Max length / stride | 8192 / 3613 |
| LR / weight decay / grad clip | `1e-5` / `0.01` / `1.0` |
| Batch | 2 × 4 accumulation (effective 8) |
| Epochs / warmup / seed | 15 / 0.2 (10-epoch reference) / 42 |
| Precision | bf16 |
| Class weights | O 0.07, B 5.0, I 1.0, E 5.0 |
| Loss | weighted token CE |
| Selection | val IoU@0.5 F1, early stopping 4 epochs |
| Best checkpoint | step 6562 |

## Test result

Held-out test (19 books). Decoder picked on validation, then frozen (Viterbi break penalty 0).

| Metric | Test |
|---|---:|
| IoU@0.5 F1 | **0.853** |
| Precision | 0.831 |
| Recall | 0.876 |
| Pred / gold spans | 2423 / 2299 |

**Viterbi + IoU@0.5.** The head emits a score per token per label. Argmax can jump to an illegal BIOE tag (e.g. `I` with no `B`). Viterbi picks the highest-scoring *legal* path (`B I* E`). A predicted span is a hit if it overlaps a gold span by IoU ≥ 0.5 — at least half the union of the two token ranges — then precision/recall/F1 are micro-averaged over those span matches.

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).
