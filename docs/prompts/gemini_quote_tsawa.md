# Gemini prompt — QUOTE / TSAWA span extraction

Built 2026-09-03 from the gold layers in `Data/repos` (45 Tsawa-bearing books,
9,586 Quotation spans, 3,687 Tsawa spans). Every statistic quoted inside the
prompt was measured on that sample; the worked examples are copied verbatim from
the corpus.

Send everything between the two rulers as a single user turn, with `«chunk»`
replaced by one window of text. Temperature 0.

---

You are a philologist of classical Tibetan Buddhist literature, working on
commentaries (བསྟན་བཅོས་ / འགྲེལ་པ་) transcribed from woodblock prints. You are given
one chunk of running text. Mark the spans in it that belong to one of two
categories, and nothing else.

## The two categories

**QUOTE** — words taken from *another* work and reproduced inside this one: a
sūtra, a tantra, a śāstra, an earlier master's writing, or an utterance
attributed to a named speaker. The author has stopped speaking and is citing.

**TSAWA** (རྩ་བ, "root text") — the words of the *base text that this document is
a commentary on*, lifted out in order, piece by piece, so that the commentary can
unpack them. Nearly always verse.

Everything else is unlabelled: the commentator's own exposition, outline
headings, glosses, section numbering, colophons.

## What a QUOTE looks like

Introduced by a frame naming the source:

- `<work title> + ལས།` — མདོ་ལས། · རྒྱུད་ལས། · མཛོད་ལས། · རྒྱན་ལས། · འགྲེལ་ལས། · དེ་ཉིད་ལས། · ལུང་ལས།
- `ཇི་སྐད་དུ།` · `སྐད་དུ།` · `<work>་དུ།`
- `<person> + ཞལ་ནས།` · `ན་རེ།` · `གསུངས་པ།`
- chained citations: `... ཞེས་དང༌། ...` linking two passages under one frame

Closed by `ཞེས་` / `ཅེས་` plus one of གསུངས་སོ · སོ · དང༌ · པ་ལྟར · པ་སྟེ · འབྱུང · སོགས,
or a bare `གསུངས།`.

**Treat every one of these as permission to look, never as proof.** Measured over
15 million characters of this corpus, `ལས།` occurs about once every 1,400
characters and only a minority of those occurrences open a quotation. `ཇི་སྐད་དུ།`
— the cleanest opener there is — still opens a quotation less than half the time.
The marker tells you where to look; what settles it is whether the voice changes.

### The ལས trap

ལས is the single most misleading token in this task, because three different
things are spelled with it:

1. **ལས།** — with a *shad*, right after a work title, an author, or དེ་ཉིད:
   this is the ablative "from", and it is a citation frame. `མདོ་ལས།`,
   `སྤྱོད་འཇུག་ལས།`, `ཡིད་བཞིན་རིན་པོ་ཆེའི་མཛོད་ལས།`
2. **ལས།** — with a shad, but after a *number, quantity, or ཙམ/དགོས་པ*: this is
   partitive ("of the three, the first") or adversative ("merely X, yet Y"). It
   opens nothing. `རྒྱུ་མཚན་གསུམ་ལས། དང་པོ།` · `ཁམས་གསུམ་ལས། འདོད་པའི་སས་བསྡུས་...` ·
   `བཏགས་པ་ཙམ་ལས། མཚན་དཔེའི་ཆོས་སོགས་...` · `རྙེད་དགོས་པ་ལས། རྡུལ་ཙམ་ཡང་མི་དམིགས་པས...`
   In the gold data this shape opens a quotation **six times less often** than
   the title shape does.
3. **ལས་** — with a *tsheg*, not a shad: this is the noun ལས (action, karma) or a
   mid-phrase ablative. `ལས་ཀྱི་གཏུམ་མོ` · `ལས་དང་པོ་པ`. It is roughly three times
   more frequent than ལས། and it is **never** a citation frame. Do not let it
   trigger anything.

So: ལས earns attention only when a shad follows it *and* a work or a person
stands before it. The same caution applies to ནས and དུ.

Length: median ~110 characters, ninetieth percentile ~270, and a long tail past
1,000. Prose quotations from sūtras are common and long; do not assume a
quotation is short.

## What a TSAWA looks like

- Introduced by the commentary's own outline, ending in **ནི།** followed by a
  line break: `དང་པོ་༼མདོར་བསྡུས་ཏེ་བསྟན་པ་༽ནི།` · `བཅུ་པ་༼...༽ནི།` · `གསུམ་པ་...ནི།`.
  **62%** of gold TSAWA spans have ནི in the 30 characters before them; **38%**
  sit immediately after the closing ༽ of an outline heading.
- Verse: **97%** contain a line break. Lines close with `། །`, and a stanza
  number (༣༧, ༡༡༧) may follow the last line.
- Followed by `ཞེས་པ་སྟེ།` · `ཅེས་པ་སྟེ།` · `ཞེས་པས།` · `ཅེས་ཚིགས་བཅད་...ཀྱིས་བསྟན་ཏེ།` ·
  `ཞེས་སོ།` — and then the commentary explains those very words, often re-using
  them phrase by phrase.
- Successive TSAWA spans in one document continue one another: they are
  consecutive pieces of a single root text, not independent citations.

## Telling them apart

|  | QUOTE | TSAWA |
|---|---|---|
| whose words | some *other* work | *the* work being commented on |
| introduced by | source name + ལས།/ནས།, ཇི་སྐད་དུ། | outline topic + ནི། |
| ལས།/ནས།/སྐད་དུ before it | ~28% | ~1% |
| ནི before it | ~2% | ~62% |
| verse layout | ~36% | ~97% |
| its job | to prove or support a point | to be the thing explained |

Decision procedure:

1. Is there a named source before it (a work title, an author, ཇི་སྐད་དུ)? → QUOTE,
   even if it is verse.
2. Is it introduced by an outline topic ending in ནི།, and does the prose right
   after it gloss its own words? → TSAWA.
3. Neither, but the register is canonical and the surrounding text is commentary
   discussing it? → QUOTE.
4. If the chunk shows no root-text / commentary alternation anywhere, TSAWA is
   almost certainly absent. Do not invent one.

## Boundaries

- A span is the **quoted words only**. The introducing frame (`...ལས།`, `ནི།`) and
  the closing particle (`ཞེས་གསུངས་སོ།`) stay **outside** it.
- Begin at the first syllable of the quoted material and end at its last.
- Keep everything inside exactly as printed: tsheg ་, shad །, double shad ། །,
  line breaks, stanza numbers.
- Mark the whole quotation, not a sample of it. Several stanzas cited together
  under one frame are one span. A chain `A ཞེས་དང༌། B` sharing a single source
  frame is one span running from the start of A to the end of B.
- If a quotation is cut by the edge of the chunk, mark the part that is present.

## Do not mark

- The commentator's own argument, even when it ends ...ཡིན་ནོ། / ...ཕྱིར་རོ། /
  ...ཞེས་པ་ལྟར་རོ།
- Outline headings and enumerations: དང་པོ་ནི། · གཉིས་པ་ལ་གསུམ། · ༼...༽ནི།
- A citation frame or closer on its own. `ཞེས་གསུངས་སོ།` is never a span.
- A work title mentioned without quoted words following it — that is a reference,
  not a quotation.
- A partitive or adversative ལས།: `...གསུམ་ལས། དང་པོ།`, `...ཙམ་ལས།`. Nothing is being
  cited; the sentence is still the commentator's.
- Colophons, printing notes, and dedications.

Most chunks contain no span at all, and `{"spans": []}` is the right answer more
often than not. Do not use it as a hedge, though: an unframed canonical passage
sitting inside commentary is still a QUOTE.

## Worked examples

**1 — QUOTE, framed and closed**

Text: `...བཙལ་ནས་སླར་དམིགས་པ་ཤོར་ཏེ་འགྲོ་བས། དབུས་མཐའ་རྣམ་འབྱེད་དུ། ལེ་ལོ་དང་ནི་གདམས་ངག་རྣམས། །བརྗེད་དང་བྱིང་དང་རྒོད་པ་དང༌། །འདུ་མི་བྱེད་དང་འདུ་བྱེད་དེ། །དེ་དག་ཉེས་པ་ལྔར་འདོད་དོ། །ཞེས་གསུངས། དེ་སྤོང་བའི་...`

```json
{"spans":[{"label":"QUOTE","text":"ལེ་ལོ་དང་ནི་གདམས་ངག་རྣམས། །བརྗེད་དང་བྱིང་དང་རྒོད་པ་དང༌། །འདུ་མི་བྱེད་དང་འདུ་བྱེད་དེ། །དེ་དག་ཉེས་པ་ལྔར་འདོད་དོ།"}]}
```

The title `དབུས་མཐའ་རྣམ་འབྱེད་དུ།` and the closer `ཞེས་གསུངས།` are outside the span.

**2 — QUOTE, a chain under one frame**

Text: `...རྐྱེན་དང་ལྡན་པ་ཡིན་པ་ཞིག་དགོས་ཏེ། ཇི་སྐད་དུ། དང་པོར་མ་ཡིན་ཉིད་དག་ནི། ནད་པའི་ཕྱིར། ཞེས་དང༌། ནད་པའི་ཆེད་དུ་ཡོངས་སུ་ལོངས་སྤྱད་པའི་ཕྱིར། ཞེས་གསུངས་སོ། །དེས་མཚོན་ནས་...`

```json
{"spans":[{"label":"QUOTE","text":"དང་པོར་མ་ཡིན་ཉིད་དག་ནི། ནད་པའི་ཕྱིར། ཞེས་དང༌། ནད་པའི་ཆེད་དུ་ཡོངས་སུ་ལོངས་སྤྱད་པའི་ཕྱིར"}]}
```

Two cited fragments joined by `ཞེས་དང༌།` under the single frame `ཇི་སྐད་དུ།` — one
span, and the internal `ཞེས་དང༌།` stays inside it.

**3 — TSAWA, outline topic then root verse**

Text: `...དང་པོ་༼མདོར་བསྡུས་ཏེ་བསྟན་པ་༽ནི།\nལུང་ནི་ངོ་བོའི་སྒོ་ནས་རྣམ་པ་བཅུས། །\nལེགས་གསུངས་བཀའ་དང་དགོངས་འགྲེལ་བསྟན་བཅོས་གཉིས། །\nདེ་ལ་ལུང་གི་དམ་པའི་ཆོས་ཀྱང་ངོ་བོའི་སྒོ་ནས་དབྱེ་ན་རྣམ་པ་བཅུས་ལེགས་པར་གསུངས་པའི་བཀའ་དང༌། བཀའི་དགོངས་པ་འགྲེལ་པའི་བསྟན་བཅོས་གཉིས་ཏེ། ལྷའི་བུས་ཞུས་པའི་མདོ་ལས། ཆོས་རྣམས་ཐམས་ཅད་བཀའ་དང་བསྟན་བཅོས་གཉིས་སུ་འདུས། །ལེགས་པར་གསུངས་དང་དེ་ཡི་དགོངས་འགྲེལ་བ། །ཞེས་གསུངས་སོ། །...`

```json
{"spans":[{"label":"TSAWA","text":"ལུང་ནི་ངོ་བོའི་སྒོ་ནས་རྣམ་པ་བཅུས། །\nལེགས་གསུངས་བཀའ་དང་དགོངས་འགྲེལ་བསྟན་བཅོས་གཉིས། "},{"label":"QUOTE","text":"ཆོས་རྣམས་ཐམས་ཅད་བཀའ་དང་བསྟན་བཅོས་གཉིས་སུ་འདུས། །ལེགས་པར་གསུངས་དང་དེ་ཡི་དགོངས་འགྲེལ་བ། །"}]}
```

One real window, both labels. The first two verse lines follow an outline heading
ending in `ནི།` and the prose immediately after re-uses their words — TSAWA. The
second pair of lines is fetched from a named sūtra (`ལྷའི་བུས་ཞུས་པའི་མདོ་ལས།`) to
back the point up, and closed with `ཞེས་གསུངས་སོ།` — QUOTE. Both frames and the
closer stay outside the spans.

**4 — TSAWA again, so the pattern is unmistakable**

Text: `...བཏུལ་བར་མཛད་དོ། །\nབཅུ་པ་༼མངོན་པར་སངས་རྒྱས་པའི་མཛད་པ་༽ནི།\nབསམ་གཏན་བདེན་བཞི་རྟེན་འབྲེལ་རིམ་གཟིགས་ནས། །\nརྡོ་རྗེ་ལྟ་བུས་མངོན་པར་རྫོགས་བྱང་ཆུབ། །\nནམ་གྱི་གུང་ལ་བསམ་གཏན་བཞི་པོར་རིམ་གྱིས་སྙོམས་པར་ཞུགས། ལྷའི་སྤྱན་སོགས་མངོན་པར་ཤེས་པས་བདེན་པ་བཞི་དང༌།...`

```json
{"spans":[{"label":"TSAWA","text":"བསམ་གཏན་བདེན་བཞི་རྟེན་འབྲེལ་རིམ་གཟིགས་ནས། །\nརྡོ་རྗེ་ལྟ་བུས་མངོན་པར་རྫོགས་བྱང་ཆུབ། "}]}
```

The tenth topic of an outline, its two root lines, then the commentary walking
through them (བསམ་གཏན་བཞི་པོར... བདེན་པ་བཞི་དང... རྟེན་འབྲེལ...). No source is named,
so this is not a QUOTE.

**5 — nothing to mark**

Text: `ཉལ་འགྲོ་འདུག་ལ་སོགས་པའི་སྤྱོད་ལམ་ཅི་བྱེད་ཀྱང༌། སྔོན་དུ་རང་གི་སེམས་ཀྱི་ཀུན་སློངས་ལ་བརྟགས་ཤིང་དཔྱད་ཅིང༌། རྒྱུན་དུ་དྲན་པ་དང་ཤེས་བཞིན་ལྡན་པའི་སྒོ་ནས་གཞན་དོན་སྒྲུབ་པར་བྱེད་པ་ནི་རྒྱལ་སྲས་བྱང་ཆུབ་སེམས་དཔའི་ཉམས་ལེན་ཡིན་ནོ།།`

```json
{"spans": []}
```

Commentator's own prose. It ends in ...ཡིན་ནོ།། and contains ནི, but nothing here is
another text's words.

## Output

Return the spans in the order they appear in the chunk. Do not return character
offsets, indices, translations, explanations, or markdown fences.

Reply with JSON only: `{"spans":[{"label":"QUOTE"|"TSAWA","text":"..."}]}`
Each `text` copied character for character. If none: `{"spans": []}`.
Text: «chunk»

---
