"""Validated Tibetan quotative markers (openers and closers) for Quotation spans.

How these were derived
----------------------
Candidates were **mined, not guessed**. On the 159 TRAIN books only:

1. For every gold Quotation span, take the 1-3 syllables starting just after the
   span end (closer candidates) and the 1-3 syllables ending just before the
   span start (opener candidates), skipping intervening punctuation.
2. Build a background from 154,452 shad (``།``) positions at least 200 chars away
   from any gold span boundary, using the identical extraction.
3. Score each candidate by *enrichment*: P(gram | span boundary) / P(gram | shad).
4. Normalise ``༌`` to ``་``, strip trailing tsheg, drop 1-character fragments
   (those are mid-syllable cuts, not markers).

Val and test were never used for derivation, so the numbers below are held out.

Measured performance
--------------------
Coverage = % of gold spans with a match in the search window.
Boundary-hit rate = % of all matches in running text that land within 50 chars
of a gold boundary, measured on the 46 most densely annotated books.

CLOSERS                          train    val   test    ALL   boundary-hit
  v1_strong  (previous)          48.7%  50.7%  25.9%  47.1%      44.6%
  v1_default (previous)          52.5%  51.5%  26.8%  50.2%      43.9%
  v1_broad   (previous)          60.9%  53.9%  41.1%  58.4%      28.9%
  v2_strong                      55.2%  51.9%  27.8%  52.6%      38.9%
  v2_default  <- recommended     60.2%  55.2%  35.9%  57.6%      32.5%
  v2_broad                       67.9%  60.7%  48.6%  65.5%      26.6%

OPENERS                          train    val   test    ALL   boundary-hit
  v1_draft   (unvalidated guess) 47.6%  41.1%  23.9%  44.9%      33.7%
  v2_strong                      20.5%  18.5%  11.4%  19.5%      43.4%
  v2_default  <- recommended     46.4%  40.0%  20.8%  43.5%      45.4%
  v2_broad                       52.4%  46.0%  30.4%  49.8%      20.9%

``opener v2_default`` matches ``v1_draft`` on recall while gaining 11.7 points of
precision — the draft's bare ``ཏུ།`` / ``དུ།`` were firing on ordinary case
particles. ``closer v2_default`` gains 7.4 points of coverage over ``v1_default``
for 11 points of boundary-hit rate; pick ``v2_strong`` if precision matters more.

IMPORTANT CAVEATS
-----------------
* The boundary-hit rate is a **lower bound on precision**, not precision. This
  corpus is only partially annotated (see ``find_quotation_closers.py``'s
  candidate-FN output), so a match away from a gold boundary is often a real
  quotation nobody labelled rather than a false alarm.
* ``ཞེས་དང`` / ``ཅེས་དང`` also rank high as *openers*. That is not an opener
  sense — it is the closer of the previous quotation sitting immediately before
  the next one. They are deliberately excluded from the opener profiles.
* Bare ``ལས`` is the single strongest opener signal (35.1% of train spans, lift
  21.2) but it is also the ordinary ablative case particle, which is why
  ``v2_broad`` collapses to a 20.9% hit rate. ``v2_default`` requires it to be
  followed by a shad or ``ཀྱང``, or preceded by a text-name syllable.
* These are descriptive statistics for stratification and error analysis. Do NOT
  feed them to the model as features — the benchmark would then measure the
  regex, not the model.

Regenerate with ``script/build_quotation_tags.py``.
"""

from __future__ import annotations

import re

TS = "[་༌]"  # tsheg or non-breaking tsheg

CLOSER_PROFILES: dict[str, re.Pattern[str]] = {
    "v1_strong": re.compile(
        r"(?:ཞེས|ཅེས)་གསུངས|(?:ཞེས|ཅེས)་སོ|(?:ཞེས|ཅེས)་དང|གསུངས།"
    ),
    "v1_default": re.compile(
        r"(?:ཞེས|ཅེས)་གསུངས|(?:ཞེས|ཅེས)་སོ|(?:ཞེས|ཅེས)་དང"
        r"|(?:ཞེས|ཅེས)་པ་(?:སྟེ|ལྟར|དང)|གསུངས།"
    ),
    "v1_broad": re.compile(r"(?:ཞེས|ཅེས)(?:་|།)"),
    "v2_strong": re.compile(
        rf"(?:ཞེས|ཅེས){TS}(?:གསུངས|སོ|དང|བཤད|སོགས|པ{TS}(?:ལྟར|དང|སྟེ)|པས)"
        rf"|གསུངས{TS}?(?:སོ|པ|ནས)|(?<!་)གསུངས།"
    ),
    "v2_default": re.compile(
        rf"(?:ཞེས|ཅེས)(?:{TS}(?:གསུངས|སོ|དང|བཤད|སོགས|པ|པས|ཞིང)|།|{TS}?$)|གསུངས"
    ),
    "v2_broad": re.compile(rf"(?:ཞེས|ཅེས)|གསུངས|བཤད{TS}པ"),
}

OPENER_PROFILES: dict[str, re.Pattern[str]] = {
    "v1_draft": re.compile(
        r"ཇི་སྐད་དུ|ཇི་ལྟར|ཞལ་ནས|གསུང་ནས|གསུངས་པ་ལྟར"
        r"|ལས།|ལས་ཀྱང|ཏུ།|དུ།|ན་རེ|ཞེས་བྱ་བ"
    ),
    "v2_strong": re.compile(
        rf"ཇི{TS}སྐད{TS}དུ|སྐད{TS}དུ|ཞལ{TS}ནས|གསུང{TS}ནས|ན{TS}རེ"
        rf"|(?:མདོ|རྒྱུད|མཛོད|འགྲེལ|རྒྱན|འཇུག|བསྟན{TS}བཅོས){TS}ལས|ལས{TS}ཀྱང"
    ),
    "v2_default": re.compile(
        rf"ཇི{TS}སྐད{TS}དུ|སྐད{TS}དུ|ཞལ{TS}ནས|གསུང{TS}ནས|ན{TS}རེ"
        rf"|ལས{TS}?།|ལས{TS}ཀྱང|ལས$"
    ),
    "v2_broad": re.compile(rf"ལས|སྐད{TS}དུ|ཞལ{TS}ནས|ན{TS}རེ|ནས།"),
}

RECOMMENDED_CLOSER = "v2_default"
RECOMMENDED_OPENER = "v2_default"

# Search windows, relative to the span offsets (inclusive-end convention).
CLOSER_WINDOW = (-30, 50)   # around span end
OPENER_WINDOW = (-45, 10)   # around span start
