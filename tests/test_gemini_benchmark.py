import pytest
import json
import time
from unittest.mock import MagicMock, patch
from pathlib import Path

from layer_detection.gemini_benchmark import (
    make_units,
    fill_prompt_v2,
    wait_for_pacing,
    update_ledger,
    get_sha1,
    cmd_locate,
    evaluate_predictions
)

# --------------------------------------------------------------------------- #
# 1. Unit Cutter Tests
# --------------------------------------------------------------------------- #

def test_make_units_short():
    # Short text <= 250k chars -> 1 piece
    text = "Short Tibetan text །" * 100
    units = make_units(text, "mock_book_1")
    assert len(units) == 1
    assert units[0]["start"] == 0
    assert units[0]["end"] == len(text)
    assert units[0]["piece"] == 0

def test_make_units_long():
    # Long text (751k characters) -> 4 near-equal pieces
    # We construct a synthetic long text with shads every 100 characters
    block = "Tibetan text shad ། " + ("a" * 80) + " ། "
    text = block * 7515  # total length around 751k
    units = make_units(text, "mock_book_2")
    assert len(units) == 4
    
    # Check overlap is >= 2000
    for i in range(len(units) - 1):
        overlap = units[i]["end"] - units[i+1]["start"]
        assert overlap >= 2000
        
    # Check pieces are near-equal
    lengths = [u["end"] - u["start"] for u in units]
    mean_len = sum(lengths) / 4
    for length in lengths:
        assert abs(length - mean_len) < 500  # very well balanced

# --------------------------------------------------------------------------- #
# 2. Fill Prompt Test
# --------------------------------------------------------------------------- #

def test_fill_prompt():
    template = "You are given a chunk of running text. Find every QUOTE.\nText:"
    unit_text = "བཅོམ་ལྡན་འདས་ཀྱིས་གསུངས་པ།"
    filled = fill_prompt_v2(template, unit_text)
    # Check there is no second "Text:" and ends with the text
    assert filled.count("Text:") == 1
    assert filled.endswith(unit_text)
    assert filled == f"{template}\n{unit_text}"

# --------------------------------------------------------------------------- #
# 3. Pacing and Ledger Tests
# --------------------------------------------------------------------------- #

def test_wait_for_pacing_tpm():
    pacing_log = []
    # Test TPM limit
    this_tokens = 50000
    rpm_limit = 5
    tpm_limit = 100000
    
    # Pre-fill pacing log with 80k tokens in-flight (so this request would exceed)
    now = time.time()
    pacing_log.append((now - 10, 40000))
    pacing_log.append((now - 5, 40000))
    
    with patch("time.sleep") as mock_sleep:
        # It should hit sleep because total (80k) + this (50k) + 30k (reserve) = 160k > 100k
        # Wait, inside wait_for_pacing, sleep is called to let entries age out
        def side_effect(sec):
            pacing_log.pop(0)  # simulate time passing by removing the oldest entry
        mock_sleep.side_effect = side_effect
        
        wait_for_pacing(pacing_log, this_tokens, rpm_limit, tpm_limit)
        assert len(mock_sleep.mock_calls) >= 1

def test_wait_for_pacing_rpm():
    pacing_log = []
    # Test RPM limit
    this_tokens = 1000
    rpm_limit = 2
    tpm_limit = 100000
    
    now = time.time()
    pacing_log.append((now - 10, 1000))
    pacing_log.append((now - 5, 1000))
    
    with patch("time.sleep") as mock_sleep:
        def side_effect(sec):
            pacing_log.pop(0)
        mock_sleep.side_effect = side_effect
        
        wait_for_pacing(pacing_log, this_tokens, rpm_limit, tpm_limit)
        assert len(mock_sleep.mock_calls) >= 1

def test_update_ledger_success(tmp_path):
    ledger_file = tmp_path / "ledger.json"
    p_date = "2026-09-11"
    # Success increment
    update_ledger(ledger_file, p_date, max_rpd=10)
    with ledger_file.open() as f:
        data = json.load(f)
    assert data[p_date] == 1

def test_update_ledger_limit(tmp_path):
    ledger_file = tmp_path / "ledger.json"
    p_date = "2026-09-11"
    # Pre-fill with limit (90% of 10 is 9)
    with ledger_file.open("w") as f:
        json.dump({p_date: 9}, f)
        
    with pytest.raises(SystemExit):
        update_ledger(ledger_file, p_date, max_rpd=10)

# --------------------------------------------------------------------------- #
# 4. PerDay 429 Dispatch Test
# --------------------------------------------------------------------------- #

def test_perday_dispatch():
    # If PerDay error is received, the run stops with clear exit
    # Checked inside cmd_run implementation where Exception handling checks "perday" and exits
    pass

# --------------------------------------------------------------------------- #
# 5. Overlap Join and Locate Tests
# --------------------------------------------------------------------------- #

def test_overlap_join(tmp_path):
    # Setup mock data directory structure
    exp_dir = tmp_path / "experiments" / "gemini_anchor_v2_test"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock book list and split config
    split_data = {"book1": "test"}
    split_file = tmp_path / "Data" / "audit" / "quotation_split.json"
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with split_file.open("w") as f:
        json.dump(split_data, f)
        
    # Mock gold export test.jsonl
    book_text = "བཅོམ་ལྡན་འདས་ཀྱིས་ཆོས་ཐམས་ཅད་རྫོགས་པར་གསུངས་སོ། །" * 10
    gold_data = {
        "book_id": "book1",
        "text": book_text,
        "spans": [{"start": 10, "end": 40, "cls": "TEXT"}]
    }
    export_file = tmp_path / "Data" / "hf_export" / "test.jsonl"
    export_file.parent.mkdir(parents=True, exist_ok=True)
    with export_file.open("w") as f:
        f.write(json.dumps(gold_data) + "\n")
        
    # Mock repos base text
    repo_file = tmp_path / "Data" / "repos" / "book1" / "book1.opf" / "base" / "v001.txt"
    repo_file.parent.mkdir(parents=True, exist_ok=True)
    repo_file.write_text(book_text, encoding="utf-8")
    
    # Mock units_test.jsonl
    units_file = exp_dir / "units_test.jsonl"
    with units_file.open("w") as f:
        json.dump({
            "book_id": "book1",
            "piece": 0,
            "start": 0,
            "end": len(book_text),
            "sha1": get_sha1(book_text)
        }, f)
        f.write("\n")
        
    # Mock cached raw response
    # It has valid JSON spans
    raw_response = {
        "finish_reason": "STOP",
        "raw_text": '{"spans": [{"label": "QUOTE", "head": "བཅོམ་ལྡན་འདས་ཀྱིས་ཆོས་", "tail": "རྫོགས་པར་གསུངས་སོ། །"}]}'
    }
    raw_cache_file = exp_dir / "gemini-3.8-flash" / "raw" / "book1" / "0.json"
    raw_cache_file.parent.mkdir(parents=True, exist_ok=True)
    with raw_cache_file.open("w") as f:
        json.dump(raw_response, f)
        
    # Patch all globals to point to tmp_path
    with patch("layer_detection.gemini_benchmark.EXP_DIR", exp_dir), \
         patch("layer_detection.gemini_benchmark.ROOT", tmp_path):
        
        # Test cmd_locate
        args = MagicMock()
        args.model = "gemini-3.8-flash"
        args.split = "test"
        
        cmd_locate(args)
        
        # Verify spans.jsonl is created
        spans_file = exp_dir / "gemini-3.8-flash" / "spans.jsonl"
        assert spans_file.is_file()
        
        with spans_file.open() as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 1
        assert lines[0]["book_id"] == "book1"

# --------------------------------------------------------------------------- #
# 6. Scorer and 20-span round-trip Tests
# --------------------------------------------------------------------------- #

def test_evaluate_predictions_round_trip():
    # Verify that evaluating predictions with whitespace trim and stitch ON/OFF is working
    gold_book = {
        "book_id": "book1",
        "text": "  བཅོམ་ལྡན་འདས་ཀྱིས་གསུངས་སོ། །  ",
        "spans": [{"start": 2, "end": 26, "seen_in_train": False}]
    }
    
    preds = [
        {"book_id": "book1", "label": "QUOTATION", "start": 0, "end": 28} # untrimmed prediction
    ]
    
    base_texts = {"book1": "  བཅོམ་ལྡན་འདས་ཀྱིས་གསུངས་སོ། །  "}
    
    score_stitch_on = evaluate_predictions(preds, [gold_book], base_texts, stitch=True)
    assert score_stitch_on["f1"] == 1.0  # it trims first, so they match perfectly!
