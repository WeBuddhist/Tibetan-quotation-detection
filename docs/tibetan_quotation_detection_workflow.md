# Tibetan Annotation Layer Detection — Full Training Workflow

*(Originally scoped as quotation-only; generalized to detect multiple annotation layer types in one model, since they share the same underlying task shape and rarely overlap.)*

## 0. Data summary

- Corpus: 266 OpenPecha (OPF) books, spot-checked to confirm distinct texts, not duplicates.
- Each book = raw text file (`base/v001.txt`) + YAML layers giving character-offset spans across **8 annotation types**: Quotation, Sabche, Yigchung, Tsawa, Chapter, Author, BookTitle, Citation.
- None of these are marked with punctuation in Tibetan script — they're signaled by phrases, formatting conventions, or scribal practice (e.g. quotations by "ཅེས་གསུངས," "thus said"), which is why this needs a model rather than regex.
- Per-type coverage (books / spans / invalid spans):
  - **Quotation**: 195 books / 25,210 spans / 480 invalid (1.9%)
  - **Sabche**: 138 books / 18,847 spans / 0 invalid
  - **Tsawa**: 89 books / 6,845 spans / 0 invalid
  - **Yigchung**: 119 books / 9,027 spans / 2 invalid
  - **Chapter**: 159 books / 1,667 spans / 30 invalid (1.8%)
  - **Author**: 260 books / 311 spans / 0 invalid — ~1 per book. Structurally more like document metadata than a repeating within-text span, but **kept in model scope for now** per decision below rather than dropped.
  - **BookTitle**: 266 books / 266 spans / 0 invalid — 1 per book. Same caveat as Author — kept in scope for now.
  - Citation: 2 books / 260 spans / 229 invalid (88%) — too sparse and too broken to train on; excluded.
- **Model scope: seven types** — Quotation, Sabche, Tsawa, Yigchung, Chapter, **and (for now) Author, BookTitle**. Citation remains excluded (too sparse, too broken). Author/BookTitle are ~1-per-book rather than repeating within-text spans, so they're a weaker sequence-labeling signal than the other five — revisit dropping them back to metadata-only if they underperform badly or drag down shared training (see step 8), but they are not being dropped preemptively.
- Overlap check: Author and BookTitle show **zero overlap** with every other type, including each other (confirmed in `annotation_overlap_summary.csv`), so adding them doesn't touch the priority rule in step 5 — only Quotation-vs-other overlaps exist in this corpus.
- Overlap between the five span-type layers is negligible even at 266 books: Quotation×Sabche overlaps in 12 span pairs, Quotation×Tsawa in 8, Quotation×Yigchung in 2, Chapter×Quotation in 1 — every other pair shows zero overlap. This is what makes one combined label set workable instead of five separate models.
- Quotation span length distribution (characters): median 113, p95 371, p99 742, max 4,391 — governs the window/stride sizing in step 5.
- **Task framing**: single multi-class token classification (NER-style), one shared mmBERT model, one combined BIO label set spanning all seven layer types: `O, B-QUOTE, I-QUOTE, B-SABCHE, I-SABCHE, B-TSAWA, I-TSAWA, B-YIGCHUNG, I-YIGCHUNG, B-CHAPTER, I-CHAPTER, B-AUTHOR, I-AUTHOR, B-BOOKTITLE, I-BOOKTITLE` (15 classes total).

## 1. Audit & clean raw annotation spans (Task #10 — runs first)

- Clean the 512 invalid spans (480 Quotation, 30 Chapter, 2 Yigchung) — all are the same `inverted_empty_start_eq_end_plus_1` pattern (start = end+1, i.e. a zero-width/backwards marker), none recoverable, so all 512 are **dropped** (0 fixed). Sabche/Tsawa are already clean (0 invalid).
- Exclude Citation entirely (2 books, 88% invalid). Author/BookTitle are kept as modeled targets for now (not dropped to metadata-only) — see decision in step 0.
- Spot-check the outlier spans near the max quotation length (4,391 chars) — confirm they're real long quotes, not annotation merges.
- This has to happen **before** anything downstream uses the spans — feeding invalid spans into the split's stratification stats or into training labels would corrupt both. Cleaning first means everything downstream works off correct data.
- Masking: a book not declaring a given layer (e.g. no Sabche) means that layer was deliberately assessed as not applicable to that text, not that it was left unchecked — confirmed with the data source (tsadra). So every unlabeled token can be treated as a genuine "O" negative for all seven modeled layer types, **except** the two meta-mismatch books below, which are excluded from scope instead of trusted as negatives. **Open item:** `layer_masks.csv` currently only computes `in_scope_*` columns for the original five span types — needs the same in-scope columns added for Author and BookTitle before training-time labeling (step 5) covers them too.
- **Meta/file mismatch — 2 books excluded from scope for one layer each:** `P000010` declares a Quotation layer in `meta.yml` but has no `Quotation.yml` file on disk, and `P000100` declares a Tsawa layer but has no `Tsawa.yml` file on disk. Since the file is missing, there's no real evidence the layer was actually checked (unlike a book that never declared the layer at all), so these are **not** treated as confirmed "O" negatives. Decision: flip `in_scope_Quotation`→0 for P000010 and `in_scope_Tsawa`→0 for P000100 in `layer_masks.csv` — both book/layer pairs are excluded from that layer's loss entirely, same as an undeclared layer.

## 2. Split — book-level, weighted iterative stratification (Task #1)

- Split **by whole book**, never by window or token, into train/val/test (~80/10/10) — this is the one non-negotiable rule regardless of step order, since it's what prevents a book's text (and especially overlapping windows of it) from leaking across splits.
- Computed entirely from the **cleaned raw span offsets** (step 1's output) — genuinely no tokenization anywhere in this step. Earlier drafts of this plan used a "non-O token count" as the sizing metric, which quietly required running a tokenizer just to get stratification stats — that was an unnecessary duplicate of the tokenization that happens for real at training time (step 5), so it's replaced here with **non-O character count** (`end - start` summed directly from the valid spans), which is available straight from the raw YAML files with zero tokenization. This makes tokenization a true one-time step in the whole pipeline, happening only in step 5. Character count is a fine proxy for "how much labeled content is in this book" — it's roughly proportional to token count for any given tokenizer, so it doesn't change which books land in which split.
- **Sizing metric: non-O (labeled) character count, not raw document length.** Of the corpus's 66.2M total characters, only 7.22M (10.9%) are non-O (valid spans across the 7 modeled layers), and raw book length correlates only weakly with labeled volume. The single largest book by raw length (P000152) is only lightly annotated relative to its size — sizing by raw length would let it eat outsized "volume" while contributing almost nothing to learn from. So 80/10/10 targets are computed on non-O character volume (train ≈5.77M, val ≈722k, test ≈722k), using each book's non-O character count as its weight.
- **Balance metric: book presence per layer, not character volume.** Book coverage per layer — Quotation 194, Sabche 138, Tsawa 89 (rarest), Yigchung 119, Chapter 159, Author 260, BookTitle 266 — sets an 80/10/10 target in book counts. Character-volume stratification is too noisy for Chapter/Author/BookTitle specifically: their spans are short, so together they're a small fraction of all non-O characters despite wide book coverage. "Does this split have its fair share of books containing this label" is the sturdier check for those three; character volume stays reserved for overall split sizing.
- **Algorithm — weighted iterative stratification** (Sechidis et al. 2011, extended with continuous per-book weights instead of unit weights):
  1. Initialize each split's remaining volume budget (`target_vol[s]`, from non-O characters) and remaining book-count need per layer per split (`target_count[layer][s]`, from book coverage).
  2. Repeat until every book is assigned: find the layer with the **fewest unassigned books still carrying it** — rarest evidence gets placed first (Tsawa's 89 books go before anything else), so it can't be accidentally exhausted by unrelated assignments later.
  3. Among unassigned books carrying that layer, take the **largest by non-O weight first** — placing heavy books while every split still has slack avoids a big book getting forced into an already-full split near the end.
  4. **Assign using a combined deficit score, not a lexicographic rule.** An earlier version of this algorithm assigned to whichever split had the largest remaining book-count need first, using volume only as a tie-break — running it for real showed this badly fails on volume: train ended up with 94.3% of labeled character volume instead of 80%, val/test got ~2.8-2.9% each instead of 10%, because val/test's small book-count quotas filled up fast with whatever books came up regardless of size, dumping most large/dense books into train by default. Fixed by scoring each split as `(remaining book-count need / target) + (remaining volume need / target)` — both normalized deficits summed — and assigning to whichever split scores highest. Verified this hits both targets at once: volume lands at 78.9%/11.1%/10.0% (vs. 80/10/10 target), and every layer's book count stays within 1-3 books of target.
  5. **Overshoot guard:** if that placement would push the split's volume past its budget by more than ~15%, assign to the next-best split for that layer that can absorb it instead — protects val/test specifically from one oversized book (the P000152 problem).
  6. Deduct the book's weight from that split's volume budget, decrement the book-count need for every layer it carries, remove it from the unassigned pool, and loop.
  7. Any leftover book carrying no modeled layer at all goes to whichever split has the most remaining volume budget, purely to true up overall size (shouldn't trigger here — Author/BookTitle cover 260/266 and 266/266 books).
- **Audit pass after assignment** (separate from the algorithm itself): compute realized non-O token volume and per-layer book count + token count for each split, compare against targets, and flag any split/layer combination that deviates by more than a few percentage points before locking the split and moving to step 3.

## 3. Push book-level dataset to Hugging Face Hub (Task #11)

This is the **last step before training** — everything above (audit, split) works directly on raw spans; nothing gets tokenized or windowed until step 5.

- **Schema — flat spans list, not pre-combined BIO tags:** `book_id`, `split`, raw `text`, and a `spans` list — `[{"start": int, "end": int, "label": "QUOTATION"|"SABCHE"|"TSAWA"|"YIGCHUNG"|"CHAPTER"|"AUTHOR"|"BOOKTITLE"}, ...]`. This is the standard offset/spans schema (same shape used by LabelStudio, spaCy-style exports, and most Hub NER datasets built on character offsets rather than pre-split words). Deliberately **not** pre-combined into BIO tags — that preserves the 23 genuinely overlapping spans at full fidelity and keeps the data tokenizer-agnostic, so the priority rule and window size can change later (e.g. if falling back to `bert-base-tibetan`) without needing to regenerate the saved data.
- Don't save subword token IDs as the canonical dataset — they're tied to one specific tokenizer, and would need full regeneration if we ever fall back to `bert-base-tibetan`. Raw text + spans is the only thing that needs to be precious and permanent.
- Build as a `DatasetDict` (`train`/`val`/`test`, from step 2's split) and push with `dataset_dict.push_to_hub("username/repo-name")`, which auto-converts to Parquet — no manual conversion needed.
- Add a dataset card (`README.md` with YAML frontmatter — language `bo`, task `token-classification`, license, source = tsadra/OpenPecha, and the known data-quality notes: 512 dropped invalid spans, Citation excluded, Author/BookTitle experimental scope) for long-term usability.
- **Test set discipline:** once pushed, treat the `test` split as fixed across every experiment (mmBERT, any later fallback model, hyperparameter variants) so results stay comparable. Use `val` for iteration and model selection; only evaluate on `test` for real candidate comparisons, not after every tweak.

**Implemented:** `python -m layer_detection.upload` (also `script/upload_to_hf.py`), targeting `karma689/layer_detection` (personal Hub; `bdrc/` namespace wasn't writable from this account — transfer to `BDRC/` later if desired). Reuses `audit_utils.py`'s `load_book`/`classify_span`/`in_scope_layers` rather than re-deriving span cleaning logic. Verified end-to-end with `--dry-run` (builds all 266 books, splits into 217/25/24 matching `split.csv`) and a spot-check of real span extraction (P000152's book title, using inclusive-end slicing `text[start:end+1]` — this codebase's span convention throughout, confirmed via `audit_utils.span_length = end - start + 1`).

**Project structure / reproducibility (not part of the pipeline steps, but decided alongside them):**
- **Revised once experiments/ arrived.** The earlier call was to keep everything in `script/` and skip `src/`, on the reasoning that `src/` mainly protects against a "locally-run code masks a broken `pip install`" failure mode that only matters if this gets installed by others. That stopped applying once a second import site showed up: `experiments/mmbert_base_v1/train.py` needed the same span-cleaning/split logic the upload path uses, and with more scripts coming (uploading to hf, setting up vast, more experiments), reaching into `script/` via `sys.path` hacks from every new consumer doesn't scale.
- **Now:** all reusable logic **and** pipeline CLIs live in `src/layer_detection/` (`audit.py`, `split.py`, `upload.py`, plus `audit_utils.py`, `split_utils.py`, `tokenize_utils.py`, `dataset_utils.py`), installed editable via `pip install -e .` (root-level `pyproject.toml`, `[tool.setuptools.packages.find] where = ["src"]`; console scripts `layer-audit` / `layer-split` / `layer-upload`). `script/` holds thin wrappers plus one-offs (`download_data.py`, `analyze_annotations.py`), and `experiments/<name>/` holds per-run training scripts — both import `layer_detection.*` the same way, so there's exactly one copy of any given piece of logic and no path hacks anywhere.
- What reproducibility actually needed: `requirements.txt` (dependencies pinned to exact verified versions: `transformers==5.14.1`, `datasets==5.0.1`, `huggingface_hub==1.25.1`, `numpy==2.2.6`; `torch` added once training starts), `pip install -e .` for the package, and a top-level `README.md` documenting the exact run order (`download_data.py` → `python -m layer_detection.audit` → `python -m layer_detection.split` → `python -m layer_detection.upload` → `experiments/mmbert_base_v1/train.py`). Split is deterministic (no RNG), so this plus pinned deps plus a Hub-revision-pinned dataset load at training time covers the reproducibility chain end to end.

## 4. Pretrained checkpoint — decided: mmBERT (Task #4)

**Decision: `jhu-clsp/mmBERT-base`** (Apache 2.0, 307M total params / 110M non-embedding). Chosen for its 8,192-token context (removes most chunking complexity) and tokenizer that needs no Tibetan-specific pre-segmentation.

Known risk to monitor at evaluation: mmBERT's Tibetan pretraining exposure was thin (introduced only in the final "decay phase," sourced from modern web text, not classical Buddhist commentary), and its own release notes report no clear advantage over older multilingual models on NER/POS-style tasks — the closest published proxy to this one. Fallback if it underperforms: `bert-base-tibetan` (Apache 2.0), which would reintroduce 512-token windowing.

## 5. Build the training script — tokenize, window, and train (Task #5, absorbing #2/#3)

Tokenizing and windowing are **training-time steps**, not pre-upload preprocessing — they happen here, right before the `Trainer` runs, using whatever checkpoint is active that day (mmBERT now, `bert-base-tibetan` if we ever fall back). This keeps the Hub dataset (step 3) reusable across checkpoints without regeneration.

1. **Load** the pushed dataset from the Hub (`load_dataset("username/repo-name")`) — gives back `text` + `spans` per book, per split.
2. **Convert spans to combined BIO tags** per book, applying the overlap priority rule here (not at storage time):
   - **Quotation always wins** when it overlaps any other layer — every one of the 23 observed overlaps involves Quotation on one side (10 full nesting, 13 boundary-crossing), so this rule alone resolves all of them in the current corpus.
   - **Fallback — longer span wins:** for any overlap this rule doesn't cover (e.g. two non-Quotation layers overlapping) — not observed in the current 266-book corpus, but the inference pipeline (step 9) will run on new, unseen books where it could occur — the span with the greater total character length wins.
3. **Tokenize with windowing** using mmBERT's native sliding-window support: `tokenizer(text, max_length=8192, stride=5120, return_overflowing_tokens=True, return_offsets_mapping=True)`. Handles windowing, offset tracking, and per-book origin (`overflow_to_sample_mapping`) automatically — no hand-rolled slicing code needed. Every window inherits its book's split assignment, so no window can ever cross a split boundary. Repeated tokens in each stride prefix remain as model context but receive loss label `-100`, preventing overlap from double-counting training and validation targets.
   - **Stride tuned against the real span-length distribution, not a round-number default.** The original 50% overlap (stride 4096, i.e. an 4096-token overlap between consecutive windows) was a safe placeholder. Measured against all 61,661 real spans in the corpus (char lengths converted to tokens via each book's actual tokenizer ratio): the longest span anywhere in the corpus is 2,940 tokens (a Quotation), with 99.9% under 1,131. The guarantee that matters — every span appears whole in at least one window — only needs overlap ≥ max span length. `stride: 5120` (3,072-token overlap) keeps that guarantee with margin (3,072 > 2,940) while cutting total windows corpus-wide from 10,609 to 8,567 — a 19% reduction in token-slots processed (86.9M → 70.2M), for zero spans put at risk. (A more aggressive `stride: 5632`, 2,560-token overlap, saves further — down to 73.8% of current windows — but puts exactly 2 of the 61,661 spans at risk of never appearing whole in a window; not used here since 5120 gets most of the win for free.)
4. **Align BIO labels to tokenizer tokens by character offset.** Every non-special token receives the winning span's tag; special/empty-offset tokens receive `-100`. The earlier “first subword only” wording assumed pre-tokenized words, but this pipeline intentionally sends raw Tibetan text to mmBERT, so there is no external word boundary or meaningful `word_ids()` grouping to use.
5. This whole tokenize→window→align pipeline is **disposable and regenerable** — never treat its output as the thing to save long-term; step 3's Hub dataset is the permanent artifact, this is just what gets built fresh from it each time training runs.

Model/training setup:

- `AutoModelForTokenClassification` loaded from `jhu-clsp/mmBERT-base`, **15 labels** (O + B/I for each of the 7 layer types).
- Requires a `transformers` version with ModernBERT/mmBERT support. Flash Attention 2 recommended on CUDA GPUs (Ampere+) for the long 8,192-token sequences.
- 110M non-embedding parameters (307M total, due to mmBERT's large multilingual embedding table) — fine-tunes comfortably on a single consumer/cloud GPU; longer sequences use more memory per batch than classic BERT, so batch size or gradient accumulation may need adjusting.
- **Class-weighted loss is more important now than in the single-label version** — the seven types have very different volumes (Quotation 25k spans vs. Chapter 1.7k vs. BookTitle 266), so an unweighted model risks under-serving the rarer types while overall accuracy still looks fine. Author/BookTitle at ~1 span/book are the most extreme case of this imbalance. The per-book `in_scope_layers` mask is applied to the class logits before cross-entropy, so unchecked layers contribute no negative gradient; this is required for a partially assessed multi-class corpus.
- Starting hyperparameters: learning rate 2e-5–5e-5, batch size per GPU memory, 3–5 epochs, standard weight decay — tune from validation performance.

**GPU spec (Task #6, decided ahead of the Vast.ai run):**
- **Recommended: RTX 4090, 24GB VRAM.** mmBERT-base's ModernBERT architecture (local-global alternating attention, RoPE, native Flash Attention 2 / unpadding support) is built for long context rather than brute-forcing it — static training memory (fp32 weights + gradients + AdamW's two fp32 moment buffers) is ~5GB regardless of batch size, and with `train_batch_size: 1` plus gradient checkpointing already in `config.yaml`, 24GB leaves comfortable headroom at an 8,192-token sequence length. On Vast.ai's marketplace this is the cost/performance sweet spot: ~$0.13–0.4/hr depending on host and on-demand vs. spot (RTX 3090, also 24GB Ampere, is a same-VRAM fallback if 4090 listings are scarce).
- **Faster/pricier alternative: A100 40GB**, from ~$1.09/hr — only worth it if wall-clock time matters more than cost; not required for this batch size.
- **Prerequisite fix before renting anything:** `config.yaml` currently has `fp16: false`, `bf16: false`, `attn_implementation: null` — full fp32 with standard (not Flash) attention. That pairing is wrong for 8,192-token sequences: standard attention materializes an 8192×8192 score matrix per head per layer, which is what actually blows up memory, not the model weights. Set `bf16: true` (bf16, not fp16 — no loss scaling needed, matches how mmBERT/ModernBERT was trained) and `attn_implementation: flash_attention_2`. The 24GB spec above assumes both are on.

## 6. Train a baseline (Task #6) — **done: `experiments/mmbert_base_v1/`**

- Ran the first fine-tuning pass (4 epochs, class-weighted loss, bf16 + Flash Attention 2). Evaluated with seqeval exact-match F1 and soft-F1 at tol_0 / tol_1 via `span_quality_val.json`.
- **v1 taught two training-loop bugs that must be fixed before any further tuning is meaningful** (see step 7). Hyperparameter / metric / class-weight changes are deferred to v1.3+ so each change is isolated.

**Training-loop bugs (fix in v1.2):**

- Best checkpoint was not what got saved for use. Eval loss peaked at epoch 2 (0.705) then diverged to 1.212 by epoch 4 — the saved model is the *worst* checkpoint, not the best. Two epochs of compute were wasted.
- Grad norm spikes up to 1742 during warmup (steps 0–120) caused chaotic weight updates for roughly the entire first epoch. Effective learning only stabilized around epoch 0.4. Trainer default `max_grad_norm: 1.0` was too permissive for 8192-token sequences.

**Class-weighting findings (scheduled for v1.4, not v1.2):**

- `class_weight_cap: 20.0` caused Author F1 to spike to 0.531 at epoch 1 then collapse to 0.016 by epoch 4 — the model over-rotated on rare classes each epoch.
- The `O` label contributes ~3.8M tokens vs ~554k span tokens (~7:1). Even at a low inverse-frequency weight, O-token volume dominates the gradient. Fix confirmed but deferred: pin O weight to ~0.07–0.1, span labels flat at 1.0, lower cap to 5.0.
- Do **not** drop class weighting entirely — without it the model converges to predicting O for everything (recall → 0).
- BookTitle (266 total spans) deferred to v1.5 — mask its labels to `-100` for now to remove gradient competition with Author. Author stays in.

**Metric findings (scheduled for v1.3, not v1.2):**

- Seqeval exact-match F1 (0.013) is not a useful training signal. The model has 69% interior token accuracy and Author reaches 0.39 at tol_1 — it is finding spans, just missing exact boundaries by 1–2 tokens.
- Switching `metric_for_best_model` to soft-F1 tol_1 is confirmed but deferred to v1.3 so v1.2 first establishes a clean baseline with `eval_loss` as the criterion.

## 7. v1.2 — Fix training infrastructure (exactly two changes from v1)

> v1.2 is **not** a new architecture or hyperparameter search — it is two infrastructure bug fixes that must be resolved before any further tuning is meaningful. Experiment folder: `experiments/mmbert_base_v1.2/`.

### 7.1 — Create the experiment folder

```
experiments/
  mmbert_base_v1/          ← do not touch
  mmbert_base_v1.2/        ← new
    config.yaml
    train.py               ← copy from v1 (or thin wrapper); only config changes for this run
    outputs/               ← created by train.py
    window_cache/          ← symlink or copy from v1 to avoid recompute
```

Window cache can be reused from v1 since checkpoint, max_length, stride, and scope_policy are all unchanged.

### 7.2 — `config.yaml` changes

Copy `experiments/mmbert_base_v1/config.yaml` verbatim and apply **only** the following diff:

```yaml
# CHANGE 1 — gradient clipping
# was: (not set, Trainer default of 1.0)
max_grad_norm: 0.3
# Rationale: warmup grad norm spikes reached 1742 in v1 (steps 0-120).
# 0.3 keeps early updates stable for 8192-token sequences without
# blocking legitimate large updates later in training.

# CHANGE 2 — checkpoint selection
# was: best by seqeval f1 (near-useless at 0.013) / final epoch effectively retained
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
save_strategy: epoch
eval_strategy: epoch
# Rationale: v1 eval loss peaked at epoch 2 (0.705) and diverged to 1.212
# by epoch 4. The saved v1 model is the worst checkpoint. This fix alone
# may meaningfully change final metrics without touching any hyperparameter.

# UNCHANGED — everything below stays identical to v1
experiment_name: mmbert_base_v1.2
dataset_repo: karma689/layer_detection
scope_policy: confirmed
checkpoint: jhu-clsp/mmBERT-base
attn_implementation: flash_attention_2
max_length: 8192
stride: 5120
pad_to_multiple_of: 8
learning_rate: 3.0e-5
train_batch_size: 1
eval_batch_size: 1
gradient_accumulation_steps: 8
num_train_epochs: 4
weight_decay: 0.01
warmup_ratio: 0.1
class_weighting: true
class_weight_cap: 20.0
bf16: true
gradient_checkpointing: true
seed: 42
logging_steps: 20
save_total_limit: 3        # keep 3 to see epoch 1/2/3 checkpoints side by side
report_to: [wandb]
wandb:
  project: layer-detection
  mode: online
output_dir: outputs
```

`train.py` must pass `max_grad_norm`, `load_best_model_at_end`, `metric_for_best_model`, and `greater_is_better` from config into `TrainingArguments` (v1 hardcoded best-by-`f1`; v1.2 must select by `eval_loss`).

### 7.3 — What to watch during training

Check W&B after each epoch eval. Stop early (kill the run) if:

- Eval loss rises two epochs in a row — the best checkpoint is already saved; further training is waste. With `load_best_model_at_end: true` it is safe to stop at any point.
- Grad norm in the first 50 steps exceeds 500. If it does, the 0.3 clip is not holding (possible bf16 overflow or unusually large batch) — pause and investigate before continuing.

Expected behavior vs v1:

| Metric | v1 result | v1.2 expectation |
|---|---|---|
| Grad norm (steps 0–50) | up to 1742 | should stay under 300 |
| Best eval loss | 0.705 (ep2, not saved) | 0.705 or better (now saved) |
| Saved checkpoint | epoch 4 (worst) | epoch 2 or whichever is best |
| Author F1 at best ckpt | 0.016 (ep4) | potentially ~0.531 (ep1 level) |

### 7.4 — Evaluation after training

- Run the same `span_quality_val.json` evaluation used for v1 against the **saved best checkpoint** (not `outputs/final` — use whichever checkpoint `load_best_model_at_end` restores, or the lowest-`eval_loss` `checkpoint-*` under `outputs/`).
- Compare against v1 metrics at tol_0 and tol_1. Question this run answers: *"What does v1's architecture actually produce at its best checkpoint, with stable gradient updates?"*
- If soft-F1 tol_1 on the primary classes (Quote, Sabche, Chapter) does not improve over v1's final checkpoint, the architecture or data is the bottleneck and escalate changes sooner. If it does improve, proceed to v1.3 on schedule.
- Push best checkpoint to Hub as `karma689/mmbert-base-layer-detection-v1.2`.

### 7.5 — Per-type reporting (unchanged intent from original step 7)

- Token-level accuracy remains misleading given class imbalance. Report **span-level precision/recall/F1** separately for each layer type, plus combined/micro-averaged score — both exact-match and soft/partial-overlap F1. Author/BookTitle F1 is still the signal for the "drop back to metadata" decision in step 0.

## 8. Planned subsequent experiments (do not implement until prior eval lands)

Each step waits for the previous one's eval results. **One change per run.**

### 8.1 — v1.3: Fix eval metric (one change)

- Switch `metric_for_best_model` from `eval_loss` to soft-F1 tol_1. Requires implementing a `compute_metrics` callback that returns `eval_soft_f1_tol1`.
- Everything else identical to v1.2.

### 8.2 — v1.4: Fix class weighting (one change)

- Lower `class_weight_cap` from 20.0 to 5.0.
- Pin O label weight explicitly to ~0.07 (= 554k/3800k × 0.5) after inverse-frequency computation, before applying cap.
- Mask BookTitle labels to `-100` (out of loss, not out of label set).
- Everything else identical to v1.3.

### 8.3 — v1.5: Tune learning rate (one change)

- Lower `learning_rate` from 3e-5 to 1e-5 and `warmup_ratio` from 0.1 to 0.2.
- Expect slower convergence but later overfitting onset and more stable Author/BookTitle F1. May need more than 4 epochs — set `num_train_epochs: 10` with early stopping patience of 3.

### 8.4 — Architecture / data changes (after stable v1.x baseline)

Only after the v1.x series produces a stable, well-measured baseline. Candidates: window size reduction, fallback checkpoint (`bert-base-tibetan`), BookTitle reintroduction, inter-annotator agreement study, shadow-mode deployment testing.

Also continue classic error analysis from the original plan: false positives/negatives per layer type; quotations without clean marker phrases; spans cut across window boundaries; Sabche vs. Tsawa confusion; one type stealing capacity from another.

## 9. Build inference pipeline for new texts (Task #9)

- Script that takes new raw Tibetan text, runs it directly through the mmBERT tokenizer (no botok) → windowed inference (8,192-token windows, stride matching the trained checkpoint) → merges overlapping window predictions into character-offset spans **per layer type**.
- Output each detected type in its corresponding OpenPecha layer format (`Quotation.yml`, `Sabche.yml`, `Tsawa.yml`, `Yigchung.yml`, `Chapter.yml`, `Author.yml`, `BookTitle.yml`) so results slot directly into the existing repo structure — effectively auto-generating layers for books that were never manually annotated for a given type.

---

*This plan is tracked as a live task list. Pipeline order: audit (#10) → split (#1) → push to Hub (#11) → checkpoint decided (#4, done) → tokenize/window/train (#5, done) → baseline v1 (#6, done) → **v1.2 infrastructure fixes (#7, next)** → v1.3 soft-F1 selection → v1.4 class weighting → v1.5 LR tune → error analysis / architecture (#8) → inference (#9). Tokenizing and windowing happen at training time; the Hub dataset stays raw text + spans, reusable across checkpoints.*
