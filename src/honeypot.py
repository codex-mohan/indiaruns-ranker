"""Honeypot detection plus hard-disqualifier gate."""

from . import config as C


def honeypot_score(feat: dict) -> bool:
    """Return True if the candidate has a honeypot signature."""
    yoe = feat.get("yoe", 0)

    if yoe < C.HONEYPOT_YOE_MINIMUM_REASONABLE or yoe > C.HONEYPOT_YOE_MAXIMUM:
        return True

    if feat.get("expert_zero_duration_count", 0) >= C.HONEYPOT_MIN_EXPERT_SKILLS_FOR_FLAG:
        return True

    if (
        feat.get("zero_endorsement_zero_duration_count", 0)
        >= C.HONEYPOT_MAX_SKILLS_WITH_ZERO_ENDORSEMENTS_AND_ZERO_DURATION
    ):
        return True

    if feat.get("high_claim_zero_evidence_count", 0) >= C.HONEYPOT_MIN_EXPERT_SKILLS_FOR_FLAG:
        return True

    return False


def hard_disqualifier(feat: dict) -> tuple[bool, str]:
    """Check JD hard disqualifiers. Returns (disqualified, reason)."""

    if feat.get("is_consulting_only"):
        return True, "consulting_only"

    if feat.get("is_research_only") and feat.get("no_production_code"):
        return True, "research_no_production"

    if feat.get("is_title_chaser"):
        return True, "title_chaser"

    if feat.get("has_recent_code_gap") and feat.get("no_production_code"):
        return True, "no_recent_code"

    if feat.get("closed_source_only"):
        return True, "closed_source_only"

    retrieval = feat.get("retrieval_must_count", 0) + feat.get("retrieval_nice_count", 0)
    ml = feat.get("ml_support_count", 0)
    if retrieval == 0 and ml == 0:
        return True, "no_relevant_skills"

    return False, ""


def gate(feat: dict) -> int:
    """Return 1 if candidate passes all gates, 0 otherwise."""
    if honeypot_score(feat):
        return 0
    disqualified, _ = hard_disqualifier(feat)
    if disqualified:
        return 0
    return 1
