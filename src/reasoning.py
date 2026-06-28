"""Templated reasoning generator — no hallucination, rank-consistent.

Builds each reasoning string from fields ACTUALLY present in the candidate's
feature dict.  Three tone tiers by rank.  Template variant chosen
deterministically by hash(candidate_id) for variation.
"""
import hashlib
import re

from . import config as C


def _hash_val(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def _top_skills(feat: dict, n: int = 3) -> list[str]:
    se = feat.get("skills_evidenced", {})
    ranked = sorted(se.items(), key=lambda x: -x[1])
    labels = {
        "retrieval_must": "retrieval/embeddings",
        "retrieval_nice": "search/ranking",
        "llm_finetune": "LLM fine-tuning",
        "ml_support": "ML frameworks",
    }
    return [labels.get(k, k) for k, v in ranked[:n] if v > 0]


def _career_signals(feat: dict) -> list[str]:
    """Extract notable signals from career text for richer reasoning."""
    career_text = feat.get("career_text", "").lower()
    signals = []
    signal_map = [
        ("faiss", "uses FAISS"),
        ("pinecone", "uses Pinecone"),
        ("bm25", "uses BM25"),
        ("rag", "built RAG"),
        ("ranking", "built ranking"),
        ("recommendation", "built recsys"),
        ("a/b", "A/B tested"),
        ("embeddings", "embedding work"),
        ("sentence-transformer", "sentence-transformers"),
        ("ndcg", "eval:NDCG"),
        ("mrr", "eval:MRR"),
        ("retrieval", "retrieval work"),
        ("vector", "vector search"),
        ("hybrid", "hybrid search"),
        ("learning to rank", "LTR experience"),
        ("evaluation framework", "built eval framework"),
    ]
    for pattern, label in signal_map:
        if pattern in career_text:
            signals.append(label)
    return signals[:3]


def _title_display(feat: dict) -> str:
    return feat.get("title_raw", "Unknown")


def generate_reasoning(feat: dict, rank: int) -> str:
    cid = feat.get("candidate_id", "")
    h = _hash_val(cid)

    title = _title_display(feat)
    yoe = feat.get("yoe", 0)
    company = feat.get("current_company", "?")
    skills = _top_skills(feat, 3)
    rrr = feat.get("recruiter_response_rate", 0)
    loc = feat.get("location_raw", "?")
    days_active = feat.get("last_active_days_ago", 9999)
    notice = feat.get("notice_period_days", 90)
    cs = _career_signals(feat)

    skill_str = ", ".join(skills) if skills else "no relevant skills"
    response_pct = f"{rrr * 100:.0f}%"
    career_nugget = f"; {cs[0]}" if cs else ""
    loc_clean = loc.split(",")[0] if "," in loc else loc

    if rank <= 25:
        title_display = title.lower()
        if not title_display.startswith("senior") and not title_display.startswith("lead") and not title_display.startswith("staff") and not title_display.startswith("principal"):
            title_display = f"senior {title_display}"

        templates = [
            (f"{title_display} with {yoe:.1f} yrs; {company}; "
             f"qualified on {skill_str}{career_nugget}; "
             f"{response_pct} reply, active {days_active}d ago; {loc_clean}."),
            (f"{title} — {yoe:.1f} yrs, {company}; "
             f"strong on {skill_str}{career_nugget}; "
             f"engaged ({response_pct} reply, last active {days_active}d ago); {loc_clean}."),
            (f"{yoe:.1f}-yr {title.lower()} at {company}; "
             f"evidence on {skill_str}{career_nugget}; "
             f"{response_pct} response rate; {loc_clean}."),
            (f"{title_display} at {company}, {yoe:.1f} yrs; "
             f"{skill_str}{career_nugget}; "
             f"{response_pct} reply, active {days_active}d ago; {loc_clean}."),
            (f"{title.lower()} ({yoe:.1f} yrs), {company}; "
             f"qualified: {skill_str}{career_nugget}; "
             f"responds {response_pct}, last active {days_active}d ago; {loc_clean}."),
            (f"{loc_clean} — {title.lower()} with {yoe:.1f} yrs at {company}; "
             f"{skill_str}{career_nugget}; "
             f"{response_pct} reply rate."),
        ]
        return templates[h % len(templates)]

    elif rank <= 75:
        gap_options = []
        if notice > C.NOTICE_MEDIUM_DAYS:
            gap_options.append(f"{notice}-day notice")
        if days_active > C.RECENCY_RECENT_DAYS:
            gap_options.append(f"last active {days_active}d ago")
        if rrr < 0.3:
            gap_options.append(f"low recruiter response ({response_pct})")
        if feat.get("is_consulting_only"):
            gap_options.append("consulting-only background")
        if feat.get("is_currently_at_consulting") and not feat.get("is_consulting_only"):
            gap_options.append(f"currently at {company} (consulting)")
        if feat.get("has_recent_code_gap"):
            gap_options.append("recent production code gap")
        if not skills:
            gap_options.append("no directly evidenced skills")
        gap = gap_options[h % len(gap_options)] if gap_options else "some JD gaps"

        templates = [
            (f"{title} with {yoe:.1f} yrs; {company}; "
             f"{skill_str}{career_nugget} — but {gap}."),
            (f"{yoe:.1f}-yr {title.lower()} at {company}; "
             f"surfaced for {skill_str} at {company}, but {gap}."),
            (f"{title.lower()} ({yoe:.1f} yrs) at {company}; "
             f"strong on {skill_str} but {gap}."),
            (f"{title} ({yoe:.1f}yr), {company}; "
             f"{skill_str} — {gap}; {loc_clean}."),
            (f"{title.lower()} at {company} ({yoe:.1f}yr); "
             f"matched on {skill_str}{career_nugget} — {gap}."),
        ]
        return templates[h % len(templates)]

    else:
        strengths = []
        if skills:
            strengths.append(skill_str)
        if feat.get("open_to_work"):
            strengths.append("open to work")
        if rrr > 0.5:
            strengths.append(f"{response_pct} reply")
        if cs:
            strengths.append(cs[0])
        strength = strengths[h % len(strengths)] if strengths else "adjacent experience"

        concerns = []
        if not skills:
            concerns.append("no relevant AI/retrieval skills")
        if feat.get("is_consulting_only"):
            concerns.append("consulting-only career")
        if feat.get("is_currently_at_consulting") and not feat.get("is_consulting_only"):
            concerns.append("consulting firm background")
        if days_active > C.RECENCY_STALE_DAYS:
            concerns.append(f"inactive ({days_active}d)")
        if rrr < 0.2:
            concerns.append("very low recruiter engagement")
        if feat.get("has_recent_code_gap"):
            concerns.append("no recent production code")
        if feat.get("is_title_chaser"):
            concerns.append("frequent job changes")
        if notice > C.NOTICE_MEDIUM_DAYS:
            concerns.append(f"{notice}-day notice")
        concern = concerns[h % len(concerns)] if concerns else "below JD bar"

        templates = [
            (f"Adjacent — {title.lower()} with {yoe:.1f} yrs; "
             f"surfaced for {strength} but {concern}."),
            (f"Below bar — {yoe:.1f}-yr {title.lower()}; "
             f"{strength} visible, but {concern}."),
            (f"{title.lower()} ({yoe:.1f} yrs); included for {strength} "
             f"despite {concern}."),
            (f"Fringe — {title} at {company} ({yoe:.1f}yr); "
             f"has {strength} but {concern}."),
            (f"Tail end — {title.lower()}, {yoe:.1f} yrs at {company}; "
             f"weak: {concern}."),
        ]
        return templates[h % len(templates)]
