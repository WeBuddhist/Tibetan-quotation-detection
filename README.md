# Tibetan Quotation Detection

Finding quotation spans in classical Tibetan Buddhist texts — where one text
quotes another. Two approaches live here side by side:

1. **A fine-tuned encoder.** mmBERT-base tagged with BIOE labels over long
   sliding windows (`experiments/mmbert_quotation_v3`).
2. **LLM prompting.** Gemini and Claude asked to pull quotations straight out of
   a page, then matched back to character offsets (`script/gemini_*.py`,
   `src/layer_detection/gemini_benchmark.py`).

Both are scored the same way, so the two can be compared directly. External
scoring also runs against the DharmaBench QUDT benchmark.

Write-ups with numbers: [docs/encoder-model.md](docs/encoder-model.md) and
[docs/zero-shot.md](docs/zero-shot.md).

Split out of the older `layer_detection_model_train` repo, which keeps the
seven-layer work (sabche, tsawa, yigchung, chapter, author, book title).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs src/layer_detection as an importable package
```

The package is still called `layer_detection` so every import keeps working
unchanged from the old repo.

Two API keys are needed for the LLM side:

```bash
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Layout

| Path | What's in it |
| --- | --- |
| `src/layer_detection/` | The library: windowing, BIOE labels, metrics, Gemini/Claude benchmark harness |
| `script/` | Thin CLI wrappers and one-off diagnostics |
| `script/analysis/` | `qd_*.py` — exploratory analysis of quotation markers and per-book statistics |
| `experiments/` | One folder per run: config, train script, results |
| `tests/` | pytest suite for the library |
| `docs/` | How-to and results: encoder, zero-shot, plus handoffs and prompts |
| `outputs/` | Saved eval predictions and metrics |
| `Data/` | Corpus and caches — **not in git** (see below) |

## Train the encoder

Full recipe, knobs, and numbers: [docs/encoder-model.md](docs/encoder-model.md).

You do **not** need to rebuild windows if you are training on the published
set (`karma689/tibetan-quotation-detection`, config
`windowed_w8192_s4579_bioe`, revision `w2.0`). `train.py` loads that pin and
refuses an unpinned revision.

```bash
cd experiments/mmbert_quotation_v3
python train.py --prepare-only    # data + schedule + weight checks, no GPU
python train.py --smoke           # 200 steps + one eval
python train.py                   # the run (writes outputs/, scores test)
```

GPU: a 24 GB card is enough (bf16 + Flash Attention 2; SDPA if flash-attn is
missing). On a Vast.ai box: `bash script/setup_vast_train_env.sh`.

Rebuild the windowed set only when the book-level source or the geometry
changes:

```bash
python script/download_data.py
python script/build_quotation_windows.py --config script/quotation_w8192_s4751.yaml
python script/relabel_quotation_bioe.py --push
```

## Encoder inference

Scoring a checkpoint *is* the inference path. Choose the Viterbi decode on
validation, then freeze it before looking at test.

```bash
# validation
python script/eval_quotation_v3.py \
    --config experiments/mmbert_quotation_v3/config.yaml \
    --checkpoint experiments/mmbert_quotation_v3/outputs/final \
    --split validation

# held-out test — decode tuned on validation, applied unchanged
python script/eval_quotation_v3.py \
    --config experiments/mmbert_quotation_v3/config.yaml \
    --checkpoint experiments/mmbert_quotation_v3/outputs/final \
    --split test --tune-on validation
```

`--checkpoint` can be `outputs/final`, a `checkpoint-*` folder, or the Hub id
`karma689/tibetan-quotation-detection`.

External benchmark (DharmaBench QUDT — 406 short documents, no windowing):

```bash
python script/eval_dharmabench_qudt.py \
    --config experiments/mmbert_quotation_v3/config.yaml \
    --checkpoint experiments/mmbert_quotation_v3/outputs/checkpoint-6562
```

Headline in-house result (19 test books, Viterbi, IoU@0.5): **F1 0.853**
(P 0.831, R 0.876). That number is per-window and in tokens — not the same
basis as the zero-shot table below. Details in
[docs/encoder-model.md](docs/encoder-model.md).

## Run zero-shot (Gemini / Claude)

Full method, limits, and numbers: [docs/zero-shot.md](docs/zero-shot.md).
Needs `GEMINI_API_KEY` and, for Claude, `ANTHROPIC_API_KEY`. Copy live rate
limits into `experiments/gemini_anchor_v2_test/limits.yaml` before a scored
run.

**One book** (debug / prompt work; not the quoted benchmark):

```bash
python script/gemini_extract.py --book P000152 \
    --prompt docs/prompts/gemini_quote_only_anchors_v2.md \
    --chunked --out quotes.json
python script/check_gemini_quotes.py --book P000152 --quotes quotes.json
```

**Three-model benchmark** on the 19 test books (same gold as the encoder,
character-level IoU@0.5):

```bash
python script/run_gemini_benchmark.py build-units
python script/run_gemini_benchmark.py preflight --source units
python script/run_gemini_benchmark.py freeze

python script/run_gemini_benchmark.py run --model gemini-3.8-flash --split test --source units
python script/run_gemini_benchmark.py run --model gemini-3.1-flash-lite --split test --source units
python script/run_gemini_benchmark.py run --model gemini-3.5-flash-lite --split test --source units
python script/run_gemini_benchmark.py claude-batch --action create --split test
python script/run_gemini_benchmark.py claude-batch --action poll --split test

python script/run_gemini_benchmark.py locate --model gemini-3.1-flash-lite --split test --source units
python script/run_gemini_benchmark.py locate --model gemini-3.5-flash-lite --split test --source units
python script/run_gemini_benchmark.py locate --model claude-sonnet-5 --split test --source units
python script/run_gemini_benchmark.py score --split test
```

`--split` is `test` or `validation`. `--source` is `units` (default, ~26
requests) or `windows` (16k chars, 333 requests). Writes
`experiments/gemini_anchor_v2_test/results.md`.

Latest scored test numbers (stitch on): gemini-3.1-flash-lite **F1 0.696**
(P 0.595, R 0.838); claude-sonnet-5 F1 0.454 (P 0.714, R 0.333).
gemini-3.8-flash is not scored yet.

## Data

`Data/` is gitignored — it holds a gigabyte-scale corpus and several gigabytes of
tokenized window caches, all regenerable. The quotation window set used for
v3 training is `karma689/tibetan-quotation-detection` @ `w2.0`. The older
seven-layer book export is `karma689/layer_detection`. Pin a revision when
loading either so a rerun months later cannot silently pick up different data.

`Data/repos` is a symlink to the corpus checkout in the old repo. To make this
repo fully standalone, replace the symlink with a real copy or re-run
`script/download_data.py`.

## Reproducibility notes

- Dependencies in `requirements.txt` are pinned to the exact versions this was verified against.
- `pip install -e .` is editable, so edits under `src/layer_detection/` take effect immediately.
- Spans use inclusive ends throughout: `text[start:end+1]`, and `span_length = end - start + 1`.
- Sliding-window overlap is context-only. Repeated stride-prefix tokens get label `-100` so they aren't counted twice.

## Reviewed test set (human-checked gold)

All 19 test books were reviewed span by span (accepted / edges corrected /
removed / added). The review files go in `Data/review/quotation_test/`
(`<book>-quotation.yaml`, one per book).

```bash
# build locally: book-level gold, BIOE test windows, change report (no network
# if Data/quotation_windows_w8192_s4579_bioe/shards/test is on disk)
python script/build_reviewed_benchmark.py

# upload: new configs reviewed_test + windowed_w8192_s4579_bioe_reviewed, tag w2.1
HF_TOKEN=... python script/build_reviewed_benchmark.py --push
```

The windows are the published w2.0 test windows with new labels — same input
ids, offsets and ownership mask — and the build first checks that it can
reproduce the published labels from the published gold before writing anything.

Score a checkpoint on both golds in one model pass (GPU):

```bash
python script/eval_quotation_reviewed.py \
    --config experiments/mmbert_quotation_v3/config.yaml \
    --checkpoint karma689/tibetan-quotation-detection \
    --save-logits outputs/reviewed_test_logits.npy
```

Add `--decode-from <eval_test.json>` to reuse a decode already chosen on
validation and skip the validation pass. Output: `outputs/eval_reviewed_test.json`.
