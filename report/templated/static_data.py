"""
static_data.py
Everything here is FIXED reference data (GDPR structure doesn't change) or a
deterministic threshold rule. None of this ever needs an LLM call.
"""

# ---------------------------------------------------------------------------
# 1. Chapter metadata (fixed — GDPR has exactly these 11 chapters)
# ---------------------------------------------------------------------------
CHAPTER_META = {
    1:  {"name": "General Provisions",                 "articles": "Art. 1–4"},
    2:  {"name": "Principles",                          "articles": "Art. 5–11"},
    3:  {"name": "Rights of the Data Subject",           "articles": "Art. 12–23"},
    4:  {"name": "Controller and Processor",             "articles": "Art. 24–43"},
    5:  {"name": "Third Country Transfers",              "articles": "Art. 44–49"},
    6:  {"name": "Supervisory Authorities",              "articles": "Art. 51–59"},
    7:  {"name": "Cooperation & Consistency",            "articles": "Art. 60–76"},
    8:  {"name": "Remedies, Liability & Penalties",      "articles": "Art. 77–84"},
    9:  {"name": "Special Situations",                   "articles": "Art. 85–91"},
    10: {"name": "Delegated Acts",                       "articles": "Art. 92–93"},
    11: {"name": "Final Provisions",                     "articles": "Art. 94–99"},
}

# ---------------------------------------------------------------------------
# 2. sub_id -> {article_ref, requires_text}
#    This is the master GDPR requirements checklist you already built when you
#    chunked/embedded the regulation for the RAG pipeline. Populate the FULL
#    150-item map here (or load it from a JSON/CSV file you already have —
#    see load_master_checklist() below). Sample entries below only cover the
#    sub_ids present in your test aggregator output so the pipeline runs
#    end-to-end right now.
# ---------------------------------------------------------------------------
GDPR_REQUIREMENTS_MAP = {
    "5.1.a.1":  {"article_ref": "Art. 5(1)(a)", "requires": "Personal data must be processed lawfully, fairly, and transparently."},
    "5.1.b.1":  {"article_ref": "Art. 5(1)(b)", "requires": "Data must be collected for specified, explicit, legitimate purposes and not further processed incompatibly."},
    "5.1.c.1":  {"article_ref": "Art. 5(1)(c)", "requires": "Data collected must be adequate, relevant, and limited to what is necessary (data minimisation)."},
    "5.1.d.1":  {"article_ref": "Art. 5(1)(d)", "requires": "Personal data must be accurate and kept up to date."},
    "5.1.e.1":  {"article_ref": "Art. 5(1)(e)", "requires": "Data must be kept in identifiable form no longer than necessary (storage limitation)."},
    "5.1.f.1":  {"article_ref": "Art. 5(1)(f)", "requires": "Data must be processed with appropriate security, integrity and confidentiality."},
    "6.1.a.1":  {"article_ref": "Art. 6(1)(a)", "requires": "Processing requires a valid legal basis, e.g. freely given consent."},
    "6.1.b.1":  {"article_ref": "Art. 6(1)(b)", "requires": "Processing may be based on necessity for performance of a contract."},
    "6.1.c.1":  {"article_ref": "Art. 6(1)(c)", "requires": "Processing may be based on compliance with a legal obligation."},
    "7.1.1":    {"article_ref": "Art. 7(1)",    "requires": "Consent requests must be clearly distinguishable, intelligible and easily accessible."},
    "7.1.2":    {"article_ref": "Art. 7(3)",    "requires": "Data subject must be able to withdraw consent at any time, as easily as it was given."},
    "8.1.1":    {"article_ref": "Art. 8(1)",    "requires": "Where a child's consent applies, verifiable parental/guardian authorization is required."},
    "9.1.1":    {"article_ref": "Art. 9(1)",    "requires": "Processing of special categories of personal data (health, biometric, etc.) is prohibited unless an exemption applies."},
    "12.1.1":   {"article_ref": "Art. 12(1)",   "requires": "Information must be provided in a concise, transparent, intelligible, and easily accessible form, using plain language."},
    "13.1.a.1": {"article_ref": "Art. 13(1)(a)", "requires": "The identity and contact details of the controller must be provided."},
    "13.1.c.1": {"article_ref": "Art. 13(1)(c)", "requires": "The purposes and legal basis of processing must be disclosed for each purpose."},
    "14.1.1":   {"article_ref": "Art. 14(1)",   "requires": "Where data is not obtained from the subject, equivalent transparency information must be provided."},
    "15.1.1":   {"article_ref": "Art. 15(1)",   "requires": "Data subject has the right to obtain confirmation and access to their personal data."},
    "16.1.1":   {"article_ref": "Art. 16",      "requires": "Data subject has the right to rectification of inaccurate personal data."},
    "17.1.1":   {"article_ref": "Art. 17(1)",   "requires": "Data subject has the right to erasure ('right to be forgotten') without undue delay."},
    "18.1.1":   {"article_ref": "Art. 18(1)",   "requires": "Data subject has the right to obtain restriction of processing in specified circumstances."},
    "20.1.1":   {"article_ref": "Art. 20(1)",   "requires": "Data subject has the right to receive their data in a structured, machine-readable format (portability)."},
    "21.1.1":   {"article_ref": "Art. 21(1)",   "requires": "Data subject has the right to object to processing based on legitimate interests, at any time."},
    "22.1.1":   {"article_ref": "Art. 22(1)",   "requires": "Data subject has the right not to be subject to a decision based solely on automated processing."},
    "25.1.1":   {"article_ref": "Art. 25(1)",   "requires": "Controller must implement data protection by design and by default."},
    "30.1.1":   {"article_ref": "Art. 30(1)",   "requires": "Controller must maintain a record of processing activities."},
    "32.1.1":   {"article_ref": "Art. 32(1)",   "requires": "Controller must implement appropriate technical/organizational security measures."},
    "33.1.1":   {"article_ref": "Art. 33(1)",   "requires": "A personal data breach must be notified to the supervisory authority within 72 hours."},
    "35.1.1":   {"article_ref": "Art. 35(1)",   "requires": "A Data Protection Impact Assessment (DPIA) is required for high-risk processing."},
    "36.1.1":   {"article_ref": "Art. 36(1)",   "requires": "Prior consultation with the supervisory authority is required where a DPIA indicates high residual risk."},
}


def get_requirement_meta(sub_id: str) -> dict:
    """Static lookup with a safe fallback so unmapped sub_ids never crash the pipeline."""
    if sub_id in GDPR_REQUIREMENTS_MAP:
        return GDPR_REQUIREMENTS_MAP[sub_id]
    article_num = sub_id.split(".")[0]
    return {
        "article_ref": f"Art. {article_num}",
        "requires": f"See GDPR Article {article_num} — full requirement text pending in master checklist.",
    }


def load_master_checklist(path: str) -> None:
    """
    Optional: if you already have the full 150-item requirements checklist saved
    as JSON (list of {sub_id, article_ref, requires}), call this once at startup
    to merge it into GDPR_REQUIREMENTS_MAP instead of hand-filling the dict above.
    """
    import json
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        GDPR_REQUIREMENTS_MAP[item["sub_id"]] = {
            "article_ref": item["article_ref"],
            "requires": item["requires"],
        }


# ---------------------------------------------------------------------------
# 3. Verdict display metadata
# ---------------------------------------------------------------------------
VERDICT_META = {
    "FULLY_MET":     {"label": "Fully Met",     "icon": "check",   "css": "verdict-met"},
    "PARTIALLY_MET": {"label": "Partially Met", "icon": "partial", "css": "verdict-partial"},
    "NOT_MET":       {"label": "Not Met",       "icon": "cross",   "css": "verdict-notmet"},
    "CONFLICTING":   {"label": "Conflicting",   "icon": "conflict","css": "verdict-conflict"},
"NOT_APPLICABLE": {"label": "Not Applicable", "icon": "info", "css": "verdict-na"},
}


def confidence_label(confidence: float | None) -> str:
    """Thresholds calibrated against the sample report (0.82->High, 0.71->Medium, 0.65->Medium)."""
    if confidence is None:
        return "Low"
    if confidence >= 0.8:
        return "High"
    if confidence >= 0.5:
        return "Medium"
    return "Low"


def chapter_status(score: float | None) -> str:
    if score is None:
        return "Not Assessed"
    if score < 50:
        return "High Risk"
    if score < 70:
        return "Needs Improvement"
    return "Good"


def overall_risk_level(score: float | None) -> str:
    if score is None:
        return "Not Assessed"
    if score < 50:
        return "HIGH RISK — Immediate action required"
    if score < 70:
        return "MEDIUM RISK — Action recommended"
    return "LOW RISK — Monitor and maintain"