# Workflow: Gemini 3.8 Flash, Gemini 3.1 Flash Lite and Claude Sonnet 5 on the 19 test books

Written 2026-09-11. This **replaces the model and window parts** of `results/gemini-anchor-v2-test-handoff.md` (§2, §3, §6, §8). Everything else in that handoff still holds: the §1 fixes, the one-module layout the coding agent already agreed to, the cache and ledger (§4.2–4.4), and the one scorer (§5).

**Prompt:** `docs/prompts/gemini_quote_only_anchors_v2.md`, sha1 `e88f021573bb97fd3733282caa8b4c966501bf9f`. Frozen. **Every model gets exactly the same filled text.** The file already ends with "Text:", so add the unit text after it with one newline. Don't add a second "Text:".
**Gold:** `Data/hf_export/test.jsonl` (19 books, 2,337 spans) and `validation.jsonl` (25 books, 2,005 spans).
**Encoder:** left out for now. Report 0.853 with the note "measured per window in tokens, not directly comparable".

---

## 1. The three models and their limits

| arm | model id | how it runs | limits that matter |
|---|---|---|---|
| G1 | `gemini-3.8-flash` | Gemini API, one request at a time | 5 RPM · **250K tokens/min** · **20 requests/day** |
| G2 | `gemini-3.1-flash-lite` | Gemini API, one request at a time | 15 RPM · **250K tokens/min** · 500 requests/day |
| C | `claude-sonnet-5` | **Anthropic Message Batches API** | 1M-token context · 128K max output · your account tier's limits |

- The Gemini numbers are from your AI Studio rate-limit page (2026-09-11).
- For Claude, copy your own limits from the Claude Console (Settings → Limits) into `limits.yaml` too. The runner refuses to start if an entry is missing.
- Gemini's daily reset is midnight Pacific: **12:30 IST** until 1 November 2026, then 13:30 IST.
- **Same settings everywhere, where the model allows it:**
  - temperature at the default 1.0: Google recommends it for Gemini 3, and Sonnet 5 rejects any other value;
  - thinking kept low: `thinking_level: "low"` for both Gemini models, adaptive thinking with `effort: "low"` for Claude;
  - JSON output with the schema `{spans:[{label, frame, head, tail}]}`.
  Record every setting in `run_manifest.json`.
- Keys come from the environment only, `GEMINI_API_KEY` and `ANTHROPIC_API_KEY`, and are never written to a file. Set a monthly spending limit in the Claude Console before the first batch.

---

## 2. Units: book by book, big books cut into pieces

One request per **unit**, and every model gets the same units.

- **A book of 250,000 characters or less is one unit.**
- **A longer book is cut into `ceil(length / 250,000)` near-equal pieces**, each cut at a shad, with a 2,000-character overlap. The overlap covers the prompt's 1,200-character ceiling and the locator's `--max-len 2000`.

Test set, 26 units:

| book | characters | pieces | ~Gemini tokens per piece |
|---|---:|---:|---:|
| IE3ACE254 | 751,000 | 4 | ~129K |
| I27E347E9 | 402,984 | 2 | ~138K |
| IB8D599AA | 388,143 | 2 | ~133K |
| I0B81CD66 | 347,591 | 2 | ~119K |
| I97B90DC0 | 301,570 | 2 | ~104K |
| the other 14 books | 54,308 – 247,282 | 1 each | ~40K – 166K |

Validation set: 32 units (five books are cut, into 2, 2, 2, 3 and 3 pieces).

The token figures use the rate you measured in AI Studio (about 3.0M tokens for 4,587,699 characters, so about **0.65 Gemini tokens per character**), plus about 5K for the prompt. **The biggest unit is about 166K Gemini tokens, two-thirds of the 250K per-minute cap.**

Write the units once to `units_test.jsonl` and `units_validation.jsonl` (book, piece, start, end, sha1 of the text). Every arm reads the same file.

**Why not windows everywhere:** G1 has 20 requests a day. With 16,000-character windows the test set is 333 requests, about 19 days. With these units it's 26 requests, about 2 days. Whether big units cost accuracy is checked on validation first (§5, day 1).

---

## 3. Staying under Gemini's 250K tokens per minute

The rule: **never have more than 200K tokens (80% of the cap) in flight in any 60 seconds, for each model.**

1. **Know the size before sending.** Preflight runs `count_tokens` on every filled unit, for each Gemini model, and saves it. The runner uses that number plus a **reserve of 30K for the reply and thinking**, because we don't know for certain whether Google counts output tokens against the cap.
2. **Rolling 60-second log.** Before each request, add up the tokens sent in the last 60 seconds. Send only if that total + this request + 30K ≤ 200K. Otherwise sleep until the oldest entry is more than 60 seconds old. With units of 100–170K, this means **about one request per minute** per model.
3. **A unit that is too big on its own gets refused, not sent.** Any unit over 170K counted tokens is cut in half at a shad before the run starts (preflight lists them). None are expected at 0.65 tokens per character, but the check must exist.
4. **Requests per minute** stay under 80% of RPM (4 for G1, 12 for G2). At one unit per minute, the token rule already guarantees this.
5. **Requests per day:** the per-model `ledger.json` counts every attempt and stops at 90% (18 for G1, 450 for G2). Resume after 12:30 IST.
6. **If a per-minute 429 arrives anyway:** wait for Google's `retryDelay` (or 60 s) plus jitter, then retry the same unit. If it happens twice in a row for one unit that is alone in the log, the unit is too big for the cap: cut it in half and log that. A **per-day** 429 stops the process. It never marks the unit as failed (that was bug §1.4).

G1 and G2 have separate quotas, so they run as two processes at the same time.

---

## 4. The Claude path: Message Batches API

**Why batches:** Claude's per-minute input limit depends on your account tier and may be smaller than one unit. The batch route isn't held to the normal per-minute limits (it has its own, much looser queue limits), costs half the normal price, and usually finishes within an hour (it can take up to 24 hours). With 26 units, one batch is the whole test run.

### 4.1 Preflight (free)

- Run Anthropic's `count_tokens` endpoint on every filled unit. **Claude's tokenizer is different from Gemini's**, and for Tibetan it may well be heavier (reports say Sonnet 5's tokenizer uses about 30% more tokens than older Claude models on the same text). Record Claude tokens per character next to Gemini's.
- Check that every unit fits: counted tokens + 64K for output ≤ 1M. If any doesn't, cut it (it must be cut for **all** arms, so do this before the Gemini runs start too).
- Print the **estimated cost** (input tokens × price, output estimate, batch discount) and wait for a typed "yes". Check the current Sonnet 5 price on Anthropic's pricing page, including whether requests over 200K tokens cost more.

### 4.2 Request settings

- `model: "claude-sonnet-5"`, `max_tokens: 64000` (the densest unit needs roughly 20–30K for anchors plus thinking).
- One user message holding the same filled prompt text the Gemini arms get. No system prompt, so nothing differs.
- No `temperature` (Sonnet 5 rejects it). `thinking: {type: "adaptive"}` with `effort: "low"`.
- Structured JSON output with the schema above. If a reply still comes back as broken JSON, use the same `parse_spans_json` / `salvage_spans` as for Gemini.
- `custom_id = "<book_id>__<piece>"`, so each result maps straight back to its unit.

### 4.3 Running the batch

1. Pilot: a small batch of the 2 pilot validation units (`P000118`, `I130B5D0F`). Check the whole path end to end: parse, locate, score.
2. Test: one batch of the 26 test units. Save the batch id in `claude-sonnet-5/batch.json` right after creating it, so a crash can reconnect instead of submitting again.
3. Check the batch every 60 seconds until it ends. Write each result into the same raw cache as Gemini (`<model>/raw/<book>/<piece>.json`), with `stop_reason`, token usage, and the batch id.
4. Results come back **in any order**. Always match them by `custom_id`, never by position.

### 4.4 Claude failures

| what happens | how to spot it | what to do |
|---|---|---|
| a request in the batch failed | result type `errored` | put it in a new small batch, at most 2 more tries, then save it as `blocked` |
| batch not finished in 24 h | result type `expired` | send the leftover units again in a new batch |
| reply cut off | `stop_reason: "max_tokens"` | cut the unit in half and send both halves. Do not raise the cap mid-run |
| model declined | `stop_reason: "refusal"` | save it as `blocked` with the reason, and count it in the report |
| overloaded / server error (pilot or single calls) | HTTP 529 / 500 | wait 30 → 60 → 120 → 240 s, then try again |
| rate limited (single calls) | HTTP 429 | wait for the `retry-after` header |
| bad request | HTTP 400 | do not retry. It is a config bug, so fix it |

As with Gemini: never retry because the *answer* looks bad. That would be tuning on the test set.

---

## 5. Day by day

**Day 1: set up and decide the unit size on validation**

1. The §1 fixes from the first handoff. `limits.yaml` filled in for all three models.
2. Preflight for all three models, on test and validation units. Show the table (tokens per character for each tokenizer, biggest unit, days needed, Claude cost) before anything else runs.
3. **G2 (3.1 Flash Lite) on the whole validation set, two ways:** 16,000/2,000 windows (333 requests) and units (32 requests). Also repeat the two pilot books once at both sizes, for the run-to-run noise (14 requests). About 379 requests, inside its 450.
   **Decision:** paired bootstrap over the 25 validation books. If the units' F1 is within the interval of the windows' F1, **units are the test setting for all three arms.** If windows are clearly better, the test setting is 16,000/2,000: G2 and Claude still finish quickly (Claude as a batch of 333), and G1 runs smallest book first on the common set, as the coding agent planned.
4. G1 (3.8 Flash) pilot on the two pilot units after the 12:30 IST reset (2 requests). Claude pilot batch (2 units).

**Day 2: freeze, then the test runs**

5. Write `run_manifest.json` (prompt sha1, units sha1, models, settings). Nothing changes after this.
6. G2 on the 26 test units (about 30 minutes at one unit per minute). Claude: one batch of 26. G1: the first 16 test units, smallest book first.

**Day 3: finish and score**

7. G1: the remaining 10 units.
8. Score all three with the one scorer (§6), then write `experiments/gemini_anchor_v2_test/results.md` and copy the numbers into the project.

---

## 6. Scoring: unchanged, with three arms

Everything in §5 of the first handoff applies: gold from `test.jsonl`, the in-unit locator with a forward cursor and `max-len 2000`, the three join rules for overlapping pieces, the same whitespace trim and adjacent join for every arm, the `seen_in_train == false` subset, length buckets, marker classes, per book, MUC, and the false-positive frame caveat.

Paired bootstrap (10,000 resamples over books, fixed seed) for **G1 − G2, G1 − C and G2 − C**. An interval that includes 0 means "no clear difference".

Extra rows per arm: requests, tokens in / thinking / out (each in its own tokenizer's units), blocked units, and cost for Claude.

---

## 7. Before you call it done

1. All 19 books were scored for all three arms, with total gold **2,337**, or the report says which books are missing and why.
2. All three arms used the same `units_test.jsonl` (same sha1) and the same filled prompt (same sha1).
3. G1 never went over 18 attempts in a Pacific day, and no Gemini unit was ever marked failed because of a daily 429.
4. The Claude batch id was saved before polling, and every result was matched by `custom_id`.
5. Running again from the cache makes **zero** API calls and gives byte-identical scores.
6. The `text[start:end+1]` round-trip passes on 20 random spans per arm.
