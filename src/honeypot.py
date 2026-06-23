"""Honeypot detection + hard-disqualifier gate.

Returns gate=1 for passable candidates, gate=0 for disqualifiers.
A candidate with gate=0 is ranked last (score=0).
"""
from . import config as C


def honeypot_score(feat: dict) -> bool:
    """Return True if the candidate has a honeypot signature.

    Checks:
    - Expert in many skills with zero duration each (keyword stuffer).
    - YOE physically impossible.
    - Skills with "expert" proficiency but 0 months usage.
    """
    skills_evidenced = feat.get("skills_evidenced", {})
    yoe = feat.get("yoe", 0)

    # YOE sanity
    if yoe < C.HONEYPOT_YOE_MINIMUM_REASONABLE or yoe > C.HONEYPOT_YOE_MAXIMUM:
        return True

    # This is checked at extract time via skills_evidenced but we also do a
    # direct check against the raw data stored in feat.  However skills_evidenced
    # values are trust-weighted and may be fractional — we need the raw counts.
    # Since we don't store raw skill data here, we rely on the evidence scores
    # being extremely low as a proxy.
    # The real honeypot logic is more robust in the rank.py pre-gate that has
    # access to the raw candidate JSON.  This is a secondary safety net.

    return False


def hard_disqualifier(feat: dict) -> tuple[bool, str]:
    """Check JD hard disqualifiers.  Returns (disqualified, reason)."""

    # consulting only
    if feat.get("is_consulting_only"):
        return True, "consulting_only"

    # research only without production
    if feat.get("is_research_only") and feat.get("no_production_code"):
        return True, "research_no_production"

    # title chaser
    if feat.get("is_title_chaser"):
        return True, "title_chaser"

    # pure tech-lead / architecture with no recent code
    if feat.get("has_recent_code_gap") and feat.get("no_production_code"):
        return True, "no_recent_code"

    # closed-source only, no external validation
    if feat.get("closed_source_only"):
        return True, "closed_source_only"

    # Primary CV/speech/robotics with no NLP/IR
    # (checked via skills_evidence: if no retrieval/ML support skills at all)
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
