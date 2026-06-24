"""Per-candidate feature extraction for ranking.

Builds a compact feature vector + text blob for each candidate.  The feature
vector feeds scoring.py; the text blob feeds semantic + sparse indexing.
"""
import re
from datetime import date, datetime
from typing import Any

from . import config as C

# ── helpers ────────────────────────────────────────────────────────────────
_CONSULTING_RE = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in C.CONSULTING_FIRMS) + r")\b",
    re.IGNORECASE,
)
_EVAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in C.EVAL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
_SCALE_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in C.SCALE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

_TODAY = date(2026, 6, 22)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _months_between(d1: date, d2: date) -> float:
    return (d2 - d1).days / 30.44


def _skill_alias_set() -> dict[str, str]:
    """Build a flat map: lowercase alias → canonical category."""
    m: dict[str, str] = {}
    for cat_name, groups in [
        ("retrieval_must", C.RETRIEVAL_MUST),
        ("retrieval_nice", C.RETRIEVAL_NICE),
        ("llm_finetune", C.LLM_FINETUNE),
        ("ml_support", C.ML_SUPPORT),
    ]:
        for group in groups:
            for alias in group:
                m[alias.lower()] = cat_name
    return m


_SKILL_MAP = _skill_alias_set()


# ── main extraction ────────────────────────────────────────────────────────
def extract(cand: dict) -> dict[str, Any]:
    """Return a flat feature dict for one candidate.

    Keys:
        candidate_id, yoe, yoe_band, title, title_raw,
        current_company, current_industry, current_company_size,
        is_consulting_only, has_recent_code_gap, is_title_chaser,
        is_research_only, no_production_code, closed_source_only,
        skills_evidenced: dict[str, float],  # skill_cat → evidence_score
        retrieval_must_count, retrieval_nice_count,
        llm_finetune_count, ml_support_count,
        has_eval_experience, has_scale_experience,
        location_raw, location_country, location_tier1_india,
        location_in_india, willing_to_relocate,
        notice_period_days, salary_min, salary_max,
        recruiter_response_rate, interview_completion_rate,
        offer_acceptance_rate, open_to_work,
        last_active_days_ago, profile_completeness,
        github_activity_score, skill_assessment_avg,
        text_blob: str,  # for embedding / TF-IDF
    """
    p = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})

    # ── basic ───────────────────────────────────────────────────────────
    yoe = p.get("years_of_experience", 0.0)
    title = p.get("current_title", "").strip()
    company = p.get("current_company", "").strip()
    industry = p.get("current_industry", "").strip()
    company_size = p.get("current_company_size", "")
    location_raw = p.get("location", "")
    country = p.get("country", "")
    headline = p.get("headline", "")
    summary = p.get("summary", "")

    # ── yoe band score (Gaussian) ──────────────────────────────────────
    yoe_band = max(0.0, 1.0 - ((yoe - C.YOE_PEAK) / C.YOE_SIGMA) ** 2)
    # soft wings: extend band beyond strict range
    if yoe < 3.0:
        yoe_band *= 0.4
    elif yoe < 4.0:
        yoe_band *= 0.7
    elif yoe > 12.0:
        yoe_band *= 0.5
    elif yoe > 15.0:
        yoe_band *= 0.3

    # ── consulting only check ──────────────────────────────────────────
    companies_all = [c.get("company", "") for c in career]
    consulting_count = sum(1 for c in companies_all if _CONSULTING_RE.search(c))
    is_consulting_only = (len(companies_all) > 0 and consulting_count == len(companies_all))

    # ── production code gap ────────────────────────────────────────────
    has_recent_code_gap = False
    if career:
        latest_end = None
        for c in career:
            d = _parse_date(c.get("end_date")) or _TODAY
            if latest_end is None or d > latest_end:
                latest_end = d
        if latest_end:
            months_since = _months_between(latest_end, _TODAY)
            has_recent_code_gap = months_since > 18

    # ── title chaser (avg tenure < 18 months AND progressive titles) ──
    durations = [c.get("duration_months", 0) for c in career if c.get("duration_months", 0) > 0]
    avg_tenure = sum(durations) / len(durations) if durations else 999
    is_title_chaser = False
    if len(career) >= 3 and avg_tenure < 18:
        titles_list = [c.get("title", "").lower() for c in career]
        seniority_words = ["senior", "staff", "principal", "lead", "head", "director"]
        progression = sum(1 for t in titles_list if any(w in t for w in seniority_words))
        if progression >= 3:
            is_title_chaser = True

    # ── research only (no production) ──────────────────────────────────
    titles_lower = [c.get("title", "").lower() for c in career]
    is_research_only = any(
        any(rt in t for rt in C.RESEARCH_TITLES) for t in titles_lower
    )
    non_coding_count = sum(
        1 for t in titles_lower if any(nc in t for nc in C.NON_CODING_TITLES)
    )
    no_production_code = non_coding_count >= len(titles_lower) * 0.5 and len(titles_lower) > 1

    # ── closed-source only (no external validation) ────────────────────
    closed_source_only = False
    if len(career) >= 3:
        all_large_proprietary = all(
            c.get("company_size", "") in ("5001-10000", "10001+")
            and c.get("industry", "").lower() in ("it services", "consulting")
            for c in career
        )
        # Check if candidate has certifications or publications (external validation)
        certs = cand.get("certifications", [])
        has_external = len(certs) > 0
        if all_large_proprietary and not has_external:
            closed_source_only = True

    # ── skill evidence ─────────────────────────────────────────────────
    skills_evidenced: dict[str, float] = {}
    retrieval_must_count = 0
    retrieval_nice_count = 0
    llm_finetune_count = 0
    ml_support_count = 0

    for s in skills:
        sn = s.get("name", "").lower()
        proficiency = s.get("proficiency", "beginner")
        endorsements = s.get("endorsements", 0)
        duration = s.get("duration_months", 0)

        # trust factor: penalize skills with no real usage
        proficiency_mult = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.6, "beginner": 0.3}
        pm = proficiency_mult.get(proficiency, 0.3)
        trust = pm * min(1.0, duration / 12.0) * min(1.0, endorsements / 10.0)
        # If 0 endorsements AND 0 duration → near-zero trust
        if endorsements == 0 and duration == 0:
            trust *= 0.05

        cat = _SKILL_MAP.get(sn)
        if cat:
            skills_evidenced[cat] = skills_evidenced.get(cat, 0.0) + trust
            if cat == "retrieval_must":
                retrieval_must_count += 1
            elif cat == "retrieval_nice":
                retrieval_nice_count += 1
            elif cat == "llm_finetune":
                llm_finetune_count += 1
            elif cat == "ml_support":
                ml_support_count += 1

    # ── eval + scale experience (scan descriptions) ────────────────────
    desc_text = " ".join(c.get("description", "") for c in career)
    has_eval_experience = bool(_EVAL_RE.search(desc_text))
    has_scale_experience = bool(_SCALE_RE.search(desc_text))

    # ── location ───────────────────────────────────────────────────────
    loc_lower = location_raw.lower()
    country_lower = country.lower()
    location_in_india = country_lower == "india" or any(
        city in loc_lower for city in C.LOCATION_INDIAN_CITIES
    )
    location_tier1_india = location_in_india and any(
        city in loc_lower for city in C.LOCATION_PREFERRED
    )
    willing_to_relocate = signals.get("willing_to_relocate", False)

    # ── behavioral signals ─────────────────────────────────────────────
    notice_period_days = signals.get("notice_period_days", 90)
    salary_range = signals.get("expected_salary_range_inr_lpa", {})
    salary_min = salary_range.get("min", 0.0)
    salary_max = salary_range.get("max", 0.0)
    recruiter_response_rate = signals.get("recruiter_response_rate", 0.0)
    interview_completion_rate = signals.get("interview_completion_rate", 0.0)
    offer_acceptance_rate = signals.get("offer_acceptance_rate", -1.0)
    open_to_work = signals.get("open_to_work_flag", False)
    github_activity = signals.get("github_activity_score", -1.0)
    profile_completeness = signals.get("profile_completeness_score", 0.0)
    search_appearance = signals.get("search_appearance_30d", 0)
    saved_by_recruiters = signals.get("saved_by_recruiters_30d", 0)

    # last_active recency
    last_active_str = signals.get("last_active_date", "")
    last_active_date = _parse_date(last_active_str)
    last_active_days_ago = (last_active_date - _TODAY).days if last_active_date else 9999

    # skill assessment average — only count relevant assessments
    assessments = signals.get("skill_assessment_scores", {})
    relevant_assessments = {
        k: v for k, v in assessments.items()
        if k.lower() in _SKILL_MAP or any(
            k.lower() in group
            for cat_groups in [
                C.RETRIEVAL_MUST, C.RETRIEVAL_NICE, C.LLM_FINETUNE, C.ML_SUPPORT
            ]
            for group in cat_groups
        )
    }
    skill_assessment_avg = (
        sum(relevant_assessments.values()) / len(relevant_assessments)
        if relevant_assessments else 0.0
    )
    skill_assessment_relevant_count = len(relevant_assessments)

    # ── text blob for embedding / TF-IDF ───────────────────────────────
    skill_names = " ".join(s.get("name", "") for s in skills)
    career_descs = " ".join(c.get("description", "") for c in career)
    career_titles = " ".join(c.get("title", "") for c in career)
    text_blob = " ".join(filter(None, [
        headline, summary, career_titles, skill_names, career_descs,
        industry, company,
    ]))
    career_text = " ".join(c.get("description", "") for c in career)

    return {
        "candidate_id": cand.get("candidate_id", ""),
        "yoe": yoe,
        "yoe_band": yoe_band,
        "title": _normalize_title(title),
        "title_raw": title,
        "current_company": company,
        "current_industry": industry,
        "current_company_size": company_size,
        "is_consulting_only": is_consulting_only,
        "has_recent_code_gap": has_recent_code_gap,
        "is_title_chaser": is_title_chaser,
        "is_research_only": is_research_only,
        "no_production_code": no_production_code,
        "closed_source_only": closed_source_only,
        "skills_evidenced": skills_evidenced,
        "retrieval_must_count": retrieval_must_count,
        "retrieval_nice_count": retrieval_nice_count,
        "llm_finetune_count": llm_finetune_count,
        "ml_support_count": ml_support_count,
        "has_eval_experience": has_eval_experience,
        "has_scale_experience": has_scale_experience,
        "location_raw": location_raw,
        "location_country": country,
        "location_in_india": location_in_india,
        "location_tier1_india": location_tier1_india,
        "willing_to_relocate": willing_to_relocate,
        "notice_period_days": notice_period_days,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "recruiter_response_rate": recruiter_response_rate,
        "interview_completion_rate": interview_completion_rate,
        "offer_acceptance_rate": offer_acceptance_rate,
        "open_to_work": open_to_work,
        "last_active_days_ago": last_active_days_ago,
        "profile_completeness": profile_completeness,
        "github_activity_score": github_activity,
        "skill_assessment_avg": skill_assessment_avg,
        "skill_assessment_relevant_count": skill_assessment_relevant_count,
        "search_appearance": search_appearance,
        "saved_by_recruiters": saved_by_recruiters,
        "text_blob": text_blob,
        "career_text": career_text,
    }


_TITLE_MAP = {
    "ml engineer": "ai_ml_engineer",
    "ai engineer": "ai_ml_engineer",
    "ai specialist": "ai_ml_engineer",
    "ai research engineer": "ai_ml_engineer",
    "ai research scientist": "ai_ml_engineer",
    "ai research": "ai_ml_engineer",
    "machine learning engineer": "ai_ml_engineer",
    "senior machine learning engineer": "ai_ml_engineer",
    "senior software engineer (ml)": "ai_ml_engineer",
    "data scientist": "data_scientist",
    "data analyst": "data_analyst",
    "senior data engineer": "data_engineer",
    "data engineer": "data_engineer",
    "analytics engineer": "data_engineer",
    "software engineer": "software_engineer",
    "senior software engineer": "software_engineer",
    "backend engineer": "software_engineer",
    "frontend engineer": "software_engineer",
    "full stack developer": "software_engineer",
    "java developer": "software_engineer",
    ".net developer": "software_engineer",
    "mobile developer": "software_engineer",
    "devops engineer": "devops",
    "cloud engineer": "devops",
    "qa engineer": "qa",
}


def _normalize_title(title: str) -> str:
    t = title.lower().strip()
    for pattern, cat in _TITLE_MAP.items():
        if pattern in t:
            return cat
    return "other"
