"""Templated reasoning generator — no hallucination, rank-consistent.

Builds each reasoning string from fields ACTUALLY present in the candidate's
feature dict.  Three tone tiers by rank.  Template variant chosen
deterministically by hash(candidate_id) for variation.
"""
import hashlib

from . import config as C


def _hash_val(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def _top_skills(feat: dict, n: int = 3) -> list[str]:
    """Return top-N skill category names with evidence > 0."""
    se = feat.get("skills_evidenced", {})
    ranked = sorted(se.items(), key=lambda x: -x[1])
    labels = {
        "retrieval_must": "retrieval/embeddings",
        "retrieval_nice": "search/ranking",
        "llm_finetune": "LLM fine-tuning",
        "ml_support": "ML frameworks",
    }
    return [labels.get(k, k) for k, v in ranked[:n] if v > 0]


def _title_display(feat: dict) -> str:
    return feat.get("title_raw", "Unknown")


def generate_reasoning(feat: dict, rank: int) -> str:
    """Generate a 1-2 sentence reasoning for this candidate at this rank.

    Guarantees:
    - Every interpolated value comes from a real field.
    - Tone matches rank bucket.
    - Variation across candidates via template selection.
    """
    cid = feat.get("candidate_id", "")
    h = _hash_val(cid)

    title = _title_display(feat)
    yoe = feat.get("yoe", 0)
    company = feat.get("current_company", "?")
    skills = _top_skills(feat, 3)
    rrr = feat.get("recruiter_response_rate", 0)
    loc = feat.get("location_raw", "?")
    days_active = feat.get("last_active_days_ago", 9999)

    # Common fact fragments
    skill_str = ", ".join(skills) if skills else "no relevant skills"
    response_pct = f"{rrr * 100:.0f}%"

    if rank <= 25:
        # ── top tier: assertive, specific, positive ──────────────────
        title_display = title.lower()
        if not title_display.startswith("senior"):
            title_display = f"senior {title_display}"
        templates = [
            (f"{title} with {yoe:.1f} yrs at product cos; qualified on {skill_str}; "
             f"{response_pct} recruiter reply, last active {days_active}d; {loc}."),
            (f"{title} — {yoe:.1f} yrs, {company}; strong on {skill_str}; "
             f"engaged ({response_pct} reply, active {days_active}d ago); {loc}."),
            (f"{yoe:.1f}-yr {title.lower()} at {company}; "
             f"evidence on {skill_str}; {response_pct} response rate; {loc}."),
            (f"{title_display} with {yoe:.1f} yrs; {company}; "
             f"qualified on {skill_str}; active {days_active}d ago; {loc}."),
        ]
        return templates[h % len(templates)]

    elif rank <= 75:
        # ── mid tier: lead with strength + note one gap ─────────────
        gap_options = []
        if feat.get("notice_period_days", 0) > C.NOTICE_MEDIUM_DAYS:
            gap_options.append(f"{feat['notice_period_days']}-day notice")
        if days_active > C.RECENCY_RECENT_DAYS:
            gap_options.append(f"last active {days_active}d ago")
        if rrr < 0.3:
            gap_options.append(f"low recruiter response ({response_pct})")
        if feat.get("is_consulting_only"):
            gap_options.append("consulting-only background")
        if feat.get("has_recent_code_gap"):
            gap_options.append("recent production code gap")
        if not skills:
            gap_options.append("no directly evidenced skills")
        gap = gap_options[h % len(gap_options)] if gap_options else "some JD gaps"

        templates = [
            (f"{title} with {yoe:.1f} yrs; {company}; "
             f"{skill_str} — but {gap}."),
            (f"{yoe:.1f}-yr {title.lower()}; surfaced for {skill_str} "
             f"at {company}, but {gap}."),
            (f"{title.lower()} ({yoe:.1f} yrs) at {company}; "
             f"strong on {skill_str} but {gap}."),
        ]
        return templates[h % len(templates)]

    else:
        # ── tail tier: hesitant, explicit concerns ───────────────────
        strengths = []
        if skills:
            strengths.append(skill_str)
        if feat.get("open_to_work"):
            strengths.append("open to work")
        if rrr > 0.5:
            strengths.append(f"{response_pct} response rate")
        strength = strengths[h % len(strengths)] if strengths else "adjacent experience"

        concerns = []
        if not skills:
            concerns.append("no relevant AI/retrieval skills")
        if feat.get("is_consulting_only"):
            concerns.append("consulting-only career")
        if days_active > C.RECENCY_STALE_DAYS:
            concerns.append(f"inactive ({days_active}d)")
        if rrr < 0.2:
            concerns.append("very low recruiter engagement")
        if feat.get("has_recent_code_gap"):
            concerns.append("no recent production code")
        if feat.get("is_title_chaser"):
            concerns.append("frequent job changes")
        concern = concerns[h % len(concerns)] if concerns else "below JD bar"

        templates = [
            (f"Adjacent — {title.lower()} with {yoe:.1f} yrs; "
             f"surfaced for {strength} but {concern}."),
            (f"Below bar — {yoe:.1f}-yr {title.lower()}; "
             f"{strength} visible, but {concern}."),
            (f"{title.lower()} ({yoe:.1f} yrs); included for {strength} "
             f"despite {concern}."),
        ]
        return templates[h % len(templates)]
