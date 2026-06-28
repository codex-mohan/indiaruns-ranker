import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.honeypot import gate, honeypot_score
from src.features import extract
from src.io import stream_candidates
from src.rank import _ids_sha256, _validate_artifacts, run

DATA_ROOT = ROOT.parent / "data" / "India_runs_data_and_ai_challenge"
CANDIDATES = DATA_ROOT / "candidates.jsonl"
SUBMISSION = ROOT / "codexmohan_6487.csv"


def _submission_rows():
    with SUBMISSION.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_submission_shape_order_and_scores():
    rows = _submission_rows()

    assert len(rows) == 100
    assert [int(r["rank"]) for r in rows] == list(range(1, 101))
    assert len({r["candidate_id"] for r in rows}) == 100
    assert all(
        float(rows[i]["score"]) >= float(rows[i + 1]["score"])
        for i in range(len(rows) - 1)
    )
    assert all(r["reasoning"].strip() for r in rows)


def test_submission_candidate_ids_exist_and_top100_passes_gate():
    rows = _submission_rows()
    needed = {r["candidate_id"] for r in rows}
    found = {}

    for cand in stream_candidates(str(CANDIDATES)):
        cid = cand["candidate_id"]
        if cid in needed:
            found[cid] = cand
            if len(found) == len(needed):
                break

    assert set(found) == needed

    feats = [extract(found[r["candidate_id"]]) for r in rows]
    assert sum(1 for feat in feats if gate(feat) == 0) == 0
    assert sum(1 for feat in feats if honeypot_score(feat)) == 0


def test_stream_candidates_fails_on_malformed_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"candidate_id": "CAND_0000001"}\n{bad json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        list(stream_candidates(str(bad)))


def test_rank_missing_artifacts_fails_nonzero_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing artifacts"):
        run(str(CANDIDATES), str(tmp_path / "missing-artifacts"), str(tmp_path / "out.csv"))


def test_artifact_validation_rejects_order_mismatch(tmp_path):
    ids = ["CAND_0000001", "CAND_0000002"]
    feats = [{"candidate_id": "CAND_0000002"}, {"candidate_id": "CAND_0000001"}]
    cand_embs = np.zeros((2, 3), dtype=np.float32)
    tfidf = TfidfVectorizer().fit_transform(["alpha", "beta"])

    with pytest.raises(ValueError, match="order mismatch"):
        _validate_artifacts(str(tmp_path), np.array(ids), cand_embs, tfidf, feats)


def test_artifact_validation_rejects_manifest_hash_mismatch(tmp_path):
    ids = ["CAND_0000001", "CAND_0000002"]
    feats = [{"candidate_id": cid} for cid in ids]
    cand_embs = np.zeros((2, 3), dtype=np.float32)
    tfidf = TfidfVectorizer().fit_transform(["alpha", "beta"])
    manifest = {
        "candidate_count": 2,
        "candidate_ids_sha256": _ids_sha256(list(reversed(ids))),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate_ids_sha256"):
        _validate_artifacts(str(tmp_path), np.array(ids), cand_embs, tfidf, feats)


def test_artifact_validation_accepts_matching_manifest(tmp_path):
    ids = ["CAND_0000001", "CAND_0000002"]
    feats = [{"candidate_id": cid} for cid in ids]
    cand_embs = np.zeros((2, 3), dtype=np.float32)
    tfidf = TfidfVectorizer().fit_transform(["alpha", "beta"])
    manifest = {
        "candidate_count": 2,
        "candidate_ids_sha256": _ids_sha256(ids),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    _validate_artifacts(str(tmp_path), np.array(ids), cand_embs, tfidf, feats)
