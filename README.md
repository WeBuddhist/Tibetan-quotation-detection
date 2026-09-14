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
| `docs/` | Handoffs, prompt versions, evaluation reports |
| `outputs/` | Saved eval predictions and metrics |
| `Data/` | Corpus and caches — **not in git** (see below) |

## Pipeline

```bash
# 1. Fetch the annotated corpus (OpenPecha repos listed in Data/tsadra_repo_list.txt)
python script/download_data.py

# 2. Build the quotation-only window dataset
python script/build_quotation_windows.py --config script/quotation_w8192_s4751.yaml

# 3. Relabel those windows BIOE (B/I/O/E instead of plain BIO)
python script/relabel_quotation_bioe.py

# 4. Train
cd experiments/mmbert_quotation_v3 && python train.py

# 5. Score
python script/eval_quotation_v3.py
python script/eval_dharmabench_qudt.py       # external benchmark
python script/run_gemini_benchmark.py        # Gemini + Claude comparison
```

## Data

`Data/` is gitignored — it holds a gigabyte-scale corpus and several gigabytes of
tokenized window caches, all regenerable. The canonical copy of the cleaned
dataset lives on the Hugging Face Hub (`karma689/layer_detection`); pin a
revision when loading it so a rerun months later can't silently pick up
different data.

`Data/repos` is a symlink to the corpus checkout in the old repo. To make this
repo fully standalone, replace the symlink with a real copy or re-run
`script/download_data.py`.

## Reproducibility notes

- Dependencies in `requirements.txt` are pinned to the exact versions this was verified against.
- `pip install -e .` is editable, so edits under `src/layer_detection/` take effect immediately.
- Spans use inclusive ends throughout: `text[start:end+1]`, and `span_length = end - start + 1`.
- Sliding-window overlap is context-only. Repeated stride-prefix tokens get label `-100` so they aren't counted twice.
