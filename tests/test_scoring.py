"""Unit tests for scoring components, feature extraction, and gate logic.

Tests individual functions with synthetic data — no 100K candidate file needed.
"""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import extract
from src.honeypot import gate, honeypot_score, hard_disqualifier
from src.scoring import (
    behavioral_multiplier,
    career_fit_score,
    compute_score,
    location_score,
    skill_evidence_score,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _make_candidate(
    *,
    yoe=6.0,
    title="Senior AI Engineer",
    company="TestCo",
    industry="AI/ML",
    company_size="51-200",
    location="Pune, Maharashtra, India",
    country="India",
    career=None,
    skills=None,
    signals=None,
    education=None,
):
    """Build a minimal candidate dict for testing."""
    cand = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "current_title": title,
            "current_company": company,
            "current_industry": industry,
            "current_company_size": company_size,
            "years_of_experience": yoe,
            "location": location,
            "country": country,
            "headline": "",
            "summary": "",
        },
        "career_history": career or [],
        "skills": skills or [],
        "education": education or [],
        "redrob_signals": signals or {},
    }
    return cand


def _feat(**kwargs):
    """Extract features from a synthetic candidate."""
    return extract(_make_candidate(**kwargs))


# ═══════════════════════════════════════════════════════════════════════════
# YOE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

class TestYOEComputation:
    def test_calculated_from_career_dates(self):
        """YOE should come from career dates when available."""
        career = [
            {"title": "AI Eng", "company": "A", "start_date": "2020-01-01",
             "end_date": "2023-01-01", "duration_months": 36},
            {"title": "ML Eng", "company": "B", "start_date": "2023-01-01",
             "end_date": None, "duration_months": 30},
        ]
        feat = _feat(yoe=2.0, career=career)  # reported=2.0, calculated≈6.0
        # reported < calculated → use calculated
        assert feat["yoe"] > 5.0

    def test_underreported_uses_calculated(self):
        """Typo in YOE field should be corrected to calculated value."""
        career = [
            {"title": "ML Eng", "company": "A", "start_date": "2020-06-01",
             "end_date": None, "duration_months": 60},
        ]
        feat = _feat(yoe=1.0, career=career)
        assert feat["yoe"] > 5.0  # calculated ≈ 6yr, not reported 1.0

    def test_overreported_penalized(self):
        """Claiming more YOE than career shows should be penalized."""
        career = [
            {"title": "ML Eng", "company": "A", "start_date": "2022-01-01",
             "end_date": None, "duration_months": 36},
        ]
        feat = _feat(yoe=15.0, career=career)  # reported=15, calculated≈4.5
        # Should be penalized: calculated * sqrt(calculated/reported)
        assert feat["yoe"] < 5.0
        assert feat["yoe"] > 1.0  # but not zero (floor=0.3)

    def test_no_career_history_uses_reported(self):
        """Fallback to reported YOE when no career dates."""
        feat = _feat(yoe=7.0, career=[])
        assert feat["yoe"] == 7.0

    def test_no_career_no_reported_is_zero(self):
        """Both missing → YOE = 0."""
        feat = _feat(yoe=0.0, career=[])
        assert feat["yoe"] == 0.0

    def test_matching_reported_and_calculated(self):
        """When reported ≈ calculated, use calculated (no penalty)."""
        career = [
            {"title": "ML Eng", "company": "A", "start_date": "2020-01-01",
             "end_date": "2026-01-01", "duration_months": 72},
        ]
        feat = _feat(yoe=6.0, career=career)
        assert abs(feat["yoe"] - 6.0) < 0.5


# ═══════════════════════════════════════════════════════════════════════════
# YOE BAND
# ═══════════════════════════════════════════════════════════════════════════

class TestYOEBand:
    def test_peak_yoe(self):
        """6.5yr should be at peak."""
        feat = _feat(yoe=6.5, career=[
            {"title": "AI Eng", "company": "A", "start_date": "2020-01-01",
             "end_date": None, "duration_months": 78}
        ])
        assert feat["yoe_band"] > 0.95

    def test_low_yoe_penalized(self):
        """<3yr should get heavy penalty."""
        feat = _feat(yoe=2.0, career=[
            {"title": "AI Eng", "company": "A", "start_date": "2024-01-01",
             "end_date": None, "duration_months": 24}
        ])
        assert feat["yoe_band"] < 0.3

    def test_high_yoe_penalized(self):
        """>15yr should get penalty."""
        feat = _feat(yoe=16.0, career=[
            {"title": "AI Eng", "company": "A", "start_date": "2010-01-01",
             "end_date": None, "duration_months": 192}
        ])
        assert feat["yoe_band"] < 0.5

    def test_zero_yoe(self):
        """0yr should give 0 band."""
        feat = _feat(yoe=0.0, career=[])
        assert feat["yoe_band"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# CONSULTING DETECTION
# ═══════════════════════════════════════════════════════════════════════════

class TestConsultingDetection:
    def test_all_consulting_flagged(self):
        """Career at only consulting firms → consulting_only."""
        career = [
            {"title": "ML Eng", "company": "TCS", "duration_months": 36},
            {"title": "ML Eng", "company": "Infosys", "duration_months": 24},
        ]
        feat = _feat(company="Infosys", career=career)
        assert feat["is_consulting_only"] is True
        assert feat["is_currently_at_consulting"] is True

    def test_current_consulting_with_prior_product(self):
        """Currently at consulting, prior product → not consulting_only."""
        career = [
            {"title": "ML Eng", "company": "Google", "duration_months": 36},
            {"title": "ML Eng", "company": "Genpact AI", "duration_months": 24},
        ]
        feat = _feat(company="Genpact AI", career=career)
        assert feat["is_consulting_only"] is False
        assert feat["is_currently_at_consulting"] is True

    def test_no_consulting(self):
        """All product companies → no consulting flags."""
        career = [
            {"title": "ML Eng", "company": "Google", "duration_months": 36},
            {"title": "ML Eng", "company": "Meta", "duration_months": 24},
        ]
        feat = _feat(company="Meta", career=career)
        assert feat["is_consulting_only"] is False
        assert feat["is_currently_at_consulting"] is False

    def test_genpact_detected(self):
        """Genpact should be recognized as consulting."""
        feat = _feat(company="Genpact AI", career=[
            {"title": "ML Eng", "company": "Genpact AI", "duration_months": 48}
        ])
        assert feat["is_consulting_only"] is True

    def test_hcl_detected(self):
        """HCL should be recognized as consulting."""
        feat = _feat(company="HCL Technologies", career=[
            {"title": "ML Eng", "company": "HCL Technologies", "duration_months": 36}
        ])
        assert feat["is_consulting_only"] is True


# ═══════════════════════════════════════════════════════════════════════════
# LOCATION SCORE
# ═══════════════════════════════════════════════════════════════════════════

class TestLocationScore:
    def test_pune_preferred(self):
        feat = _feat(location="Pune, Maharashtra, India", country="India")
        assert location_score(feat) == 1.0

    def test_noida_preferred(self):
        feat = _feat(location="Noida, Uttar Pradesh, India", country="India")
        assert location_score(feat) == 1.0

    def test_india_non_preferred(self):
        feat = _feat(location="Vizag, Andhra Pradesh, India", country="India")
        assert location_score(feat) == 0.7

    def test_outside_india_no_relocation(self):
        feat = _feat(location="New York, USA", country="USA",
                     signals={"willing_to_relocate": False})
        assert location_score(feat) == 0.05

    def test_outside_india_willing_to_relocate(self):
        feat = _feat(location="London, UK", country="UK",
                     signals={"willing_to_relocate": True})
        assert location_score(feat) == 0.15

    def test_delhi_preferred(self):
        feat = _feat(location="Delhi, Delhi, India", country="India")
        assert location_score(feat) == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# BEHAVIORAL MULTIPLIER
# ═══════════════════════════════════════════════════════════════════════════

class TestBehavioralMultiplier:
    def test_no_open_to_work_bonus(self):
        """open_to_work should NOT give a bonus (removed)."""
        feat_otw = _feat(signals={"open_to_work_flag": True, "last_active_date": "2026-06-01",
                                   "recruiter_response_rate": 0.5})
        feat_not = _feat(signals={"open_to_work_flag": False, "last_active_date": "2026-06-01",
                                   "recruiter_response_rate": 0.5})
        assert behavioral_multiplier(feat_otw) == behavioral_multiplier(feat_not)

    def test_high_reply_rate_bonus(self):
        """≥70% reply rate should get +0.10."""
        feat_high = _feat(signals={"recruiter_response_rate": 0.8, "last_active_date": "2026-06-01"})
        feat_low = _feat(signals={"recruiter_response_rate": 0.1, "last_active_date": "2026-06-01"})
        assert behavioral_multiplier(feat_high) > behavioral_multiplier(feat_low)

    def test_ghost_candidate_heavy_penalty(self):
        """>180 days inactive should get -0.18."""
        feat_ghost = _feat(signals={"last_active_date": "2025-01-01", "recruiter_response_rate": 0.5})
        feat_active = _feat(signals={"last_active_date": "2026-06-01", "recruiter_response_rate": 0.5})
        assert behavioral_multiplier(feat_ghost) < behavioral_multiplier(feat_active) - 0.15

    def test_active_applicant_bonus(self):
        """>=2 applications should get +0.03."""
        feat_active = _feat(signals={"applications_submitted_30d": 5, "last_active_date": "2026-06-01",
                                      "recruiter_response_rate": 0.5})
        feat_passive = _feat(signals={"applications_submitted_30d": 0, "last_active_date": "2026-06-01",
                                       "recruiter_response_rate": 0.5})
        assert behavioral_multiplier(feat_active) > behavioral_multiplier(feat_passive)

    def test_recruiter_saves_bonus(self):
        """>=3 saves should get +0.02."""
        feat_saved = _feat(signals={"saved_by_recruiters_30d": 10, "last_active_date": "2026-06-01",
                                     "recruiter_response_rate": 0.5})
        feat_not = _feat(signals={"saved_by_recruiters_30d": 0, "last_active_date": "2026-06-01",
                                   "recruiter_response_rate": 0.5})
        assert behavioral_multiplier(feat_saved) > behavioral_multiplier(feat_not)

    def test_very_low_reply_penalty(self):
        """<15% reply should get -0.12."""
        feat = _feat(signals={"recruiter_response_rate": 0.05, "last_active_date": "2026-06-01"})
        assert behavioral_multiplier(feat) < 0.95

    def test_notice_period_penalty(self):
        """>90 days notice should get -0.05."""
        feat_long = _feat(signals={"notice_period_days": 120, "last_active_date": "2026-06-01",
                                    "recruiter_response_rate": 0.5})
        feat_short = _feat(signals={"notice_period_days": 15, "last_active_date": "2026-06-01",
                                     "recruiter_response_rate": 0.5})
        assert behavioral_multiplier(feat_long) < behavioral_multiplier(feat_short)

    def test_bounded_range(self):
        """Multiplier should stay within [BEHAVIORAL_MIN, BEHAVIORAL_MAX]."""
        # Worst case: ghost + low reply + long notice
        feat_worst = _feat(signals={
            "last_active_date": "2024-01-01",
            "recruiter_response_rate": 0.01,
            "notice_period_days": 180,
            "interview_completion_rate": 0.1,
        })
        mult = behavioral_multiplier(feat_worst)
        assert mult >= 0.50  # BEHAVIORAL_MIN
        assert mult <= 1.15  # BEHAVIORAL_MAX

        # Best case: active + high reply + short notice
        feat_best = _feat(signals={
            "last_active_date": "2026-06-20",
            "recruiter_response_rate": 0.95,
            "notice_period_days": 0,
            "interview_completion_rate": 0.95,
            "offer_acceptance_rate": 0.9,
            "github_activity_score": 80,
            "applications_submitted_30d": 10,
            "saved_by_recruiters_30d": 20,
        })
        mult = behavioral_multiplier(feat_best)
        assert mult >= 0.50
        assert mult <= 1.15


# ═══════════════════════════════════════════════════════════════════════════
# GATE LOGIC
# ═══════════════════════════════════════════════════════════════════════════

class TestGate:
    def test_normal_candidate_passes(self):
        feat = _feat(yoe=6.0, career=[
            {"title": "AI Eng", "company": "Google", "duration_months": 36}
        ], skills=[
            {"name": "FAISS", "proficiency": "expert", "duration_months": 36, "endorsements": 10}
        ])
        assert gate(feat) == 1

    def test_consulting_only_gated(self):
        career = [{"title": "ML Eng", "company": "TCS", "duration_months": 36}]
        feat = _feat(career=career)
        assert gate(feat) == 0

    def test_no_relevant_skills_gated(self):
        """No retrieval or ML skills → disqualified."""
        feat = _feat(skills=[
            {"name": "Excel", "proficiency": "expert", "duration_months": 60, "endorsements": 50}
        ])
        assert gate(feat) == 0

    def test_honeypot_zero_expert_skills(self):
        """3+ expert skills with 0 duration → honeypot."""
        skills = [
            {"name": f"Skill{i}", "proficiency": "expert", "duration_months": 0, "endorsements": 5}
            for i in range(4)
        ]
        feat = _feat(skills=skills)
        assert honeypot_score(feat) is True
        assert gate(feat) == 0

    def test_title_chaser_gated(self):
        """3+ senior titles with avg tenure < 18mo → title chaser."""
        career = [
            {"title": "Senior ML Engineer", "company": "A", "duration_months": 12},
            {"title": "Staff ML Engineer", "company": "B", "duration_months": 14},
            {"title": "Principal ML Engineer", "company": "C", "duration_months": 10},
        ]
        feat = _feat(career=career)
        assert feat["is_title_chaser"] is True
        assert gate(feat) == 0


# ═══════════════════════════════════════════════════════════════════════════
# SKILL EVIDENCE
# ═══════════════════════════════════════════════════════════════════════════

class TestSkillEvidence:
    def test_strong_retrieval_skills(self):
        """FAISS expert with endorsements should give high score."""
        skills = [
            {"name": "FAISS", "proficiency": "expert", "duration_months": 60, "endorsements": 30},
            {"name": "Information Retrieval", "proficiency": "expert", "duration_months": 48, "endorsements": 20},
        ]
        feat = _feat(skills=skills)
        assert skill_evidence_score(feat) > 0.2

    def test_zero_endorsement_zero_duration_near_zero(self):
        """Skills with 0 endorsements + 0 duration should give near-zero trust."""
        skills = [
            {"name": "FAISS", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
            {"name": "Pinecone", "proficiency": "expert", "duration_months": 0, "endorsements": 0},
        ]
        feat = _feat(skills=skills)
        assert skill_evidence_score(feat) < 0.1

    def test_no_skills_zero_score(self):
        """No skills → 0 evidence."""
        feat = _feat(skills=[])
        assert skill_evidence_score(feat) == 0.0

    def test_keyword_stuffer_suppressed(self):
        """Many skills with 0 months → trust suppressed."""
        skills = [
            {"name": name, "proficiency": "expert", "duration_months": 0, "endorsements": 0}
            for name in ["FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus",
                         "BM25", "Elasticsearch", "RAG", "LoRA", "QLoRA"]
        ]
        feat = _feat(skills=skills)
        assert skill_evidence_score(feat) < 0.15


# ═══════════════════════════════════════════════════════════════════════════
# CAREER FIT
# ═══════════════════════════════════════════════════════════════════════════

class TestCareerFit:
    def test_ai_engineer_high_fit(self):
        feat = _feat(title="Senior AI Engineer")
        assert career_fit_score(feat) >= 0.6

    def test_marketing_manager_low_fit(self):
        feat = _feat(title="Marketing Manager")
        assert career_fit_score(feat) < 0.3

    def test_consulting_penalty(self):
        career = [{"title": "ML Eng", "company": "TCS", "duration_months": 36}]
        feat_consult = _feat(career=career, company="TCS")
        feat_product = _feat(title="AI Engineer")
        assert career_fit_score(feat_consult) < career_fit_score(feat_product)


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED SCORE
# ═══════════════════════════════════════════════════════════════════════════

class TestCombinedScore:
    def test_gated_candidate_zero(self):
        """Gate=0 → score=0 regardless of other components."""
        assert compute_score(0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0) == 0.0

    def test_score_bounded(self):
        """Score should be in [0, 1]."""
        score = compute_score(1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.15)
        assert 0.0 <= score <= 1.0

    def test_stronger_candidate_higher_score(self):
        """Better components → higher score."""
        weak = compute_score(1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.8)
        strong = compute_score(1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.1)
        assert strong > weak
