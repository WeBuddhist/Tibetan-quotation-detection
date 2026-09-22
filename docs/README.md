# Documentation

User-facing write-ups for the two systems in this repo. How-to commands also
live in the root [README](../README.md).

| Doc | What it covers |
|---|---|
| [encoder-model.md](encoder-model.md) | mmBERT BIOE classifier: train, infer, in-house and DharmaBench results |
| [zero-shot.md](zero-shot.md) | Gemini / Claude anchor prompt: one-book and full benchmark, scored test table |

## Other notes in this folder

Handoffs, prompt drafts, and the experiment log. Useful if you are continuing
a run, not if you just want to train or score.

| Path | What |
|---|---|
| [experiments/README.md](experiments/README.md) | Dated experiment log (v1–v2, eval-01–03) |
| [prompts/](prompts/) | Frozen Gemini prompts (`gemini_quote_only_anchors_v2.md` is the current one) |
| [dharmabench-qudt-evaluation.md](dharmabench-qudt-evaluation.md) | Longer QUDT report (v1.2 / v1.3) |
| [three-model-test-workflow.md](three-model-test-workflow.md) | Unit sizes, quotas, Claude batch rules |
| [gemini-anchor-v2-test-handoff.md](gemini-anchor-v2-test-handoff.md) | Original benchmark handoff |
| [windowed-flash-lite-run-handoff.md](windowed-flash-lite-run-handoff.md) | 16k-window re-run vs 250k units |
