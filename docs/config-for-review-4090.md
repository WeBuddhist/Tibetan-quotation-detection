# `mmbert_quotation_v3` — config for review

Single-layer (Quotation) BIO span detection on Tibetan text, mmBERT-base.
Full config: `experiments/mmbert_quotation_v3/config.yaml`.

**On a 4090, nothing here changes** — same 24 GB as the 3090 these numbers came
from, and Ada is fine with `torch==2.13.0` (cu130) and `flash-attn==2.8.3`.
Watch *system* RAM, not VRAM: 32 GB or some swap. The one bad run in the
previous generation was host-memory thrashing, not the GPU.

## The parameters that matter

```yaml
# data — pulled from the Hub, cached, never rebuilt locally
dataset_repo: karma689/layer_detection
dataset_config_name: windowed_w8192_s4916   # 8192 window, step 4916 = 40% overlap
dataset_revision: v2.0                      # pinned; unpinned loads refused
target_layer: Quotation                     # 3 labels: O / B-QUOTE / I-QUOTE

# model
checkpoint: jhu-clsp/mmBERT-base
attn_implementation: flash_attention_2      # falls back to SDPA automatically

# optimisation
learning_rate: 1.0e-5
train_batch_size: 2
gradient_accumulation_steps: 4              # effective batch 8
num_train_epochs: 15                        # ceiling, not a target
warmup_ratio: 0.2
warmup_reference_epochs: 10                 # warmup = 0.2 x steps/epoch x 10, NOT x 15
max_grad_norm: 1.0
bf16: true
tf32: true
seed: 42

# loss — plain weighted cross-entropy, nothing else
class_weighting: true
class_weight_cap: 5.0
o_class_weight: 0.07
positive_class_weight: v13                  # -> 1.0, a 14.3:1 positive:O balance

# selection and early stopping
metric_for_best_model: iou50_f1             # Viterbi decode + IoU@0.5 span F1
evals_per_epoch: 4
early_stopping_patience: 4                  # epochs -> 16 evals
in_loop_break_penalty: 2.0                  # selection only; re-swept at the end

# held-out score, once, after training, at the decode chosen on validation
evaluate_test_at_end: true
```

Resulting schedule: 6,877 train / 897 val / 1,132 test windows minus P000010,
the one book whose Quotation layer was never assessed and which is dropped
rather than trained as "no quotations here". ~859 steps/epoch, warmup ~1,718,
eval every ~214 steps. ~30 min/epoch on a 3090.

Change `wandb.project`, `wandb.mode` and `hub_model_id` to your own — they
point at mine.

## Please flag before changing

- **the loss** — weighted cross-entropy only
- **`warmup_reference_epochs: 10`** — it is not `num_train_epochs` on purpose
- **effective batch of 8**, **`dataset_revision`**, **the split**
- **`positive_class_weight: v13`** — a no-op for this layer (Quotation is the
  most common layer, so both the 15-class and 3-class formulas land on 1.0),
  but it is what prints the number instead of leaving it implied

## Reading the result

Baseline **0.373** — v1.3's Quote IoU@0.5. *Not* 0.427, which is the
seven-layer micro-average and would make a good result look like a regression.
Secondary: token recall inside gold, baseline **0.609**.

| result vs 0.373 | reading |
|---|---|
| > 0.45 | the single-layer split is the lever; commit to it |
| 0.40 – 0.45 | real improvement from removing layer interference |
| 0.37 – 0.40 | marginal; the interference story was overstated |
| < 0.373 | the split does not help; the problem is upstream |

Quotation is the only layer DharmaBench QUDT evaluates, so this run can also be
checked zero-shot against an external benchmark. v1.3 scored **IoU F1 0.1806**
there and touched 65 of 335 gold quotes.
