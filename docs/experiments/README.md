# Experiment log — Tibetan annotation layer detection

One document per experiment. Every entry records what question it was meant to answer,
what changed, what came out, and what it changed about the plan. Numbers here are copied
from run artifacts, not retyped from memory — each entry names its source files.

Use `TEMPLATE.md` for new entries. Number training runs `vN`; number evaluation-only
experiments `eval-NN`.

## Completed

| ID | Date | What | Headline result | Doc |
|---|---|---|---|---|
| v1 | ≈2026-07-31 | mmBERT-base baseline, 4 epochs, lr 3e-5 | exact span F1 **0.013** | [v1.md](v1.md) |
| v1.2 | ≈2026-08-05 | Precision stack: lr 1e-5, grad clip, class-weight cap 5, O pinned 0.07, best-ckpt restore | exact span F1 **0.078** | [v1.2.md](v1.2.md) |
| v1.3 | ≈2026-08-06 | Identical config to v1.2, re-run | exact span F1 **0.071**, soft F1 **0.095** | [v1.3.md](v1.3.md) |
| eval-01 | 2026-08-18 | Re-score v1.2/v1.3 logits: IoU matching, Viterbi decode, threshold sweeps | IoU@0.5 **0.134 → 0.427**, no retraining | [eval-01-decoding.md](eval-01-decoding.md) |
| audit-01 | 2026-08-18 | Quotation answer-key inspection | 3 books confirmed damaged, 32 flagged | [audit-01-annotation-damage.md](audit-01-annotation-damage.md) |
| eval-02 | 2026-08-21 | Score the train split | train recall **0.98** vs val 0.58 — supervision, not capacity | [eval-02-train-memorisation.md](eval-02-train-memorisation.md) |
| eval-03 | 2026-08-20 | DharmaBench QUDT, both checkpoints | MUC-type **0.653** (2nd vs published), but IoU recall **0.099** | [eval-03-dharmabench-qudt.md](eval-03-dharmabench-qudt.md) |
| v2 | 2026-08-28 | Focal + Tversky + boundary + EER on pinned `windowed_w8192_s4916` @ `v2.0` | in-loop IoU@0.5 **0.311** (step 1070); over-proposal as span/EER ramp; hypothesis rejected | [v2.md](v2.md) |

## Planned

| ID | What | Why | Status |
|---|---|---|---|
| eval-04 | Harness extensions: MUC-5 taxonomy, character-F1, per-book FP breakdown | Turns cleaning from guesswork into a ranked worklist | not started |
| audit-02 | Corpus-wide damage screen, all seven layers | Only Quotation has ever been checked | not started |
| eval-05 | Decile-1 Quotation spans grouped by book | Tests whether the un-fittable short spans are the damaged ones | not started |
| eval-06 | Validation confidence distributions | Tests the hedging hypothesis promoted by eval-02 | not started |
| eval-07 | Null-baseline floor on DharmaBench | 60% of its test docs are empty; needed before quoting any rank | not started |
| v1.4 | Re-run v1.2 config on current code, commit hash logged | Resolves the v1.2/v1.3 divergence | not started |

## Standing rules

- **Test split is unspent.** 24 books, never run. Spend once, after the answer-key question resolves.
- **Log the code commit** with every run. The v1.2/v1.3 divergence exists only because the shared library isn't versioned per experiment.
- **Pin `dataset_revision`.** v1–v1.3 used `null`. v2 pinned `v2.0`.
- **Test-split usage log.** The 24-book test split is scored only at milestones, and every use
  is recorded here with date, checkpoints and decode settings. Selection must be complete
  before it is touched, and nothing is tuned after seeing it.

| date | checkpoints scored | decode settings | recorded in |
|---|---|---|---|
| _(v1.4 milestone — pending)_ | v1.3, v1.4 | | `v1.4.md` |

- **Every training run ends with DharmaBench QUDT.** External, independently annotated, costs
  one inference pass. A run is not finished until it has an external number. Report it as
  distance above the null-baseline floor — 60% of that test set is empty documents.
- **Keep every epoch's checkpoint**, or select on the metric you actually believe. v1.2 and
  v1.3 were selected on `soft_f1_tol1` and their intermediate checkpoints are gone, so the
  question "which epoch is best under IoU@0.5?" can never be answered for them.
