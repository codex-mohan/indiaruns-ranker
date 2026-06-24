"""Final scoring formula.

Computes the hybrid score for a single candidate given pre-computed
component scores and extracted features.
"""
import math
from . import config as C


def skill_evidence_score(feat: dict) -> float:
    """Weighted sum of skill evidence across categories.

    retrieval_must counts most, then retrieval_nice, llm_finetune, ml_support.
    """
    se = feat.get("skills_evidenced", {})
    r_must = min(se.get("retrieval_must", 0.0), 3.0)   # cap at 3
    r_nice = min(se.get("retrieval_nice", 0.0), 2.0)
    llm = min(se.get("llm_finetune", 0.0), 2.0)
    ml = min(se.get("ml_support", 0.0), 2.0)

    # Normalize to [0, 1] — max possible ~ 3+2+2+2 = 9
    raw = 0.35 * r_must + 0.25 * r_nice + 0.20 * llm + 0.20 * ml
    return min(raw / 3.0, 1.0)


def career_fit_score(feat: dict) -> float:
    """Score based on title archetype + career signals."""
    title = feat.get("title", "other")

    title_scores = {
        "ai_ml_engineer": 1.0,
        "data_scientist": 0.7,
        "data_analyst": 0.5,
        "software_engineer": 0.55,
        "data_engineer": 0.6,
        "devops": 0.3,
        "qa": 0.2,
        "other": 0.15,
    }
    base = title_scores.get(title, 0.15)

    # Boost for eval + scale experience
    if feat.get("has_eval_experience"):
        base += 0.15
    if feat.get("has_scale_experience"):
        base += 0.05

    # Education tier boost (tiebreaker, not primary signal)
    tier = feat.get("education_tier", "none")
    if tier == "tier_1":
        base += 0.03
    elif tier == "tier_2":
        base += 0.02

    # Certification boost
    cert_count = feat.get("cert_relevant_count", 0)
    if cert_count >= 3:
        base += 0.03
    elif cert_count >= 1:
        base += 0.02

    # Penalty signals
    if feat.get("is_consulting_only"):
        base *= 0.1
    if feat.get("has_recent_code_gap"):
        base *= 0.7
    if feat.get("is_research_only"):
        base *= 0.6
    if feat.get("closed_source_only"):
        base *= 0.5

    return min(base, 1.0)


def location_score(feat: dict) -> float:
    """Score based on location preference."""
    if feat.get("location_tier1_india"):
        return 1.0
    if feat.get("location_in_india"):
        return 0.8
    if feat.get("willing_to_relocate"):
        return 0.6
    # Outside India, no relocation
    return 0.3


def behavioral_multiplier(feat: dict) -> float:
    """Multiplicative envelope from behavioral signals.

    Returns value in [BEHAVIORAL_MIN, BEHAVIORAL_MAX].
    """
    mult = 1.0

    # open to work
    if feat.get("open_to_work"):
        mult += 0.05

    # recency
    days = feat.get("last_active_days_ago", 9999)
    if days <= C.RECENCY_FRESH_DAYS:
        mult += 0.08
    elif days <= C.RECENCY_RECENT_DAYS:
        mult += 0.04
    elif days <= C.RECENCY_STALE_DAYS:
        mult -= 0.03
    else:
        mult -= 0.12  # ghost candidate — heavy penalty

    # recruiter response rate
    rrr = feat.get("recruiter_response_rate", 0.0)
    if rrr >= 0.7:
        mult += 0.08
    elif rrr >= 0.4:
        mult += 0.03
    elif rrr < 0.15:
        mult -= 0.05

    # interview completion rate
    icr = feat.get("interview_completion_rate", 0.0)
    if icr >= 0.8:
        mult += 0.04
    elif icr < 0.3:
        mult -= 0.04

    # offer acceptance rate
    oar = feat.get("offer_acceptance_rate", -1.0)
    if oar >= 0:
        if oar >= 0.6:
            mult += 0.03
        elif oar < 0.2:
            mult -= 0.03

    # github activity
    gh = feat.get("github_activity_score", -1.0)
    if gh > 50:
        mult += 0.03
    elif gh < 0:
        mult -= 0.02

    # skill assessment average
    saa = feat.get("skill_assessment_avg", 0.0)
    if saa >= 70:
        mult += 0.02
    elif saa >= 40:
        mult += 0.01

    # salary reasonableness
    sal_min = feat.get("salary_min", 0.0)
    sal_max = feat.get("salary_max", 0.0)
    if sal_max > 0 and sal_min < C.SALARY_MAX_REASONABLE:
        if sal_min >= C.SALARY_MIN_REASONABLE:
            mult += 0.01
    elif sal_min > C.SALARY_MAX_REASONABLE:
        mult -= 0.03

    # notice period
    np_ = feat.get("notice_period_days", 90)
    if np_ <= C.NOTICE_GOOD_DAYS:
        mult += 0.03
    elif np_ <= C.NOTICE_MEDIUM_DAYS:
        mult += 0.0
    else:
        mult -= 0.03

    return max(C.BEHAVIORAL_MIN, min(C.BEHAVIORAL_MAX, mult))


def compute_score(
    gate_val: int,
    semantic: float,
    lexical: float,
    skill_ev: float,
    career: float,
    yoe_band: float,
    loc: float,
    behav_mult: float,
) -> float:
    """Combine all components into a final score in [0, ~1]."""
    if gate_val == 0:
        return 0.0
    raw = (
        C.W_SEMANTIC * semantic
        + C.W_LEXICAL * lexical
        + C.W_SKILL_EVIDENCE * skill_ev
        + C.W_CAREER_FIT * career
        + C.W_YOE * yoe_band
        + C.W_LOCATION * loc
    )
    return max(0.0, min(1.0, raw * behav_mult))
