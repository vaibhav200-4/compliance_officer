# frontend/app.py

import json
import os
import time
from pathlib import Path
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="GDPR Compliance Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #0EA5E9;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .status-met { color: #10B981; font-weight: bold; }
    .status-partially { color: #F59E0B; font-weight: bold; }
    .status-notmet { color: #EF4444; font-weight: bold; }
    .status-conflicting { color: #DC2626; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True if hasattr(st, 'unsafe_allow_allowed_html') else True,
)

st.markdown('<div class="main-header">🛡️ GDPR Compliance Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered GDPR Privacy Policy Gap Analysis & Audit Platform</div>', unsafe_allow_html=True)
st.divider()

# Session State Initialization
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "job_status" not in st.session_state:
    st.session_state.job_status = None
if "result_data" not in st.session_state:
    st.session_state.result_data = None

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings & Range")
    start_art = st.number_input("Start Article", min_value=1, max_value=99, value=5)
    end_art = st.number_input("End Article", min_value=1, max_value=99, value=14)
    st.info("💡 Articles 5-14 contain key principles, data subject rights, and controller obligations.")

# Top Form: Upload & Submission
with st.container():
    st.subheader("📋 Upload Privacy Policy Document")
    col1, col2 = st.columns(2)

    with col1:
        company_name = st.text_input("Company Name", value="Acme Corporation", placeholder="e.g. Acme Corp")
    with col2:
        policy_name = st.text_input("Policy Name", value="Global Privacy Policy", placeholder="e.g. Privacy Notice 2026")

    uploaded_file = st.file_uploader("Choose a policy document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

    start_btn = st.button("🚀 Start Compliance Analysis", type="primary", use_container_width=True)

if start_btn:
    try:
        files = None
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

        data = {
            "company_name": company_name,
            "policy_name": policy_name,
            "start_article": str(start_art),
            "end_article": str(end_art),
        }

        resp = requests.post(f"{BACKEND_URL}/api/analyze", data=data, files=files, timeout=10)
        if resp.status_code == 202:
            res_json = resp.json()
            st.session_state.job_id = res_json["job_id"]
            st.session_state.job_status = "queued"
            st.session_state.result_data = None
            st.success(f"Analysis job started! Job ID: `{st.session_state.job_id}`")
            st.rerun()
        else:
            st.error(f"Failed to start analysis: {resp.text}")
    except Exception as e:
        st.error(f"Backend connection error: {e}")

# Polling and Progress Tracking Section
current_status = str(st.session_state.job_status).lower() if st.session_state.job_status else None
if st.session_state.job_id and current_status in ["queued", "running"]:
    st.divider()
    st.subheader("⏳ Analysis Progress")

    try:
        status_resp = requests.get(f"{BACKEND_URL}/api/analyze/{st.session_state.job_id}/status", timeout=5)
        if status_resp.status_code == 200:
            job_info = status_resp.json()
            st.session_state.job_status = job_info["status"]
            job_st_lower = str(job_info["status"]).lower()

            if job_st_lower in ["running", "queued"]:
                prog = job_info.get("progress", 50)
                st.progress(prog / 100.0)
                st.info(f"**Status:** Analysis {job_st_lower}... | **Elapsed:** {job_info.get('elapsed_seconds', 0)}s")
                time.sleep(2)
                st.rerun()
            elif job_st_lower in ["completed", "completed_with_failures"]:
                st.success("🎉 Compliance Analysis & Report Generation Completed!")
                if job_info.get("report"):
                    st.session_state.result_data = job_info["report"]
                else:
                    res_resp = requests.get(f"{BACKEND_URL}/api/analyze/{st.session_state.job_id}/result", timeout=10)
                    if res_resp.status_code == 200:
                        st.session_state.result_data = res_resp.json()
                st.rerun()
            elif job_st_lower == "failed":
                st.error(f"❌ Analysis Job Failed: {job_info.get('error')}")
    except Exception as e:
        st.warning(f"Waiting for status update... ({e})")
        time.sleep(2)
        st.rerun()

# Results Dashboard Section
if st.session_state.result_data:
    st.divider()
    st.subheader("📊 Final GDPR Compliance Report")

    res = st.session_state.result_data

    # Display Executive Summary if present
    if "executive_summary" in res:
        st.info(f"**Executive Summary:** {res['executive_summary']}")

    # Extract all groups across article results
    all_groups = []
    if "all_groups" in res and isinstance(res["all_groups"], list) and len(res["all_groups"]) > 0:
        all_groups = res["all_groups"]
    elif "article_results" in res and isinstance(res["article_results"], dict):
        for art_data in res["article_results"].values():
            if isinstance(art_data, dict):
                all_groups.extend(art_data.get("groups", []))
    elif "groups" in res:
        all_groups.extend(res.get("groups", []))

    counts = res.get("statistics") if isinstance(res.get("statistics"), dict) else {}
    if not counts:
        counts = {"MET": 0, "PARTIALLY_MET": 0, "NOT_MET": 0, "CONFLICTING": 0, "INSUFFICIENT_EVIDENCE": 0, "NOT_APPLICABLE": 0}
        for grp in all_groups:
            st_val = grp.get("status", "INSUFFICIENT_EVIDENCE")
            counts[st_val] = counts.get(st_val, 0) + 1

    total_grps = len(all_groups)
    score = float(res.get("overall_score", 0.0)) if "overall_score" in res else (
        ((counts.get("MET", 0) * 1.0 + counts.get("PARTIALLY_MET", 0) * 0.5) / total_grps * 100.0) if total_grps > 0 else 0.0
    )

    risk_level = res.get("risk_level") or (
        "🚨 CRITICAL RISK" if counts.get("CONFLICTING", 0) > 0 or counts.get("NOT_MET", 0) > 3 or score < 50 else (
            "⚠️ HIGH RISK" if counts.get("NOT_MET", 0) > 0 or score < 75 else (
                "⚡ MEDIUM RISK" if score < 90 else "✅ LOW RISK"
            )
        )
    )

    # Score Metrics Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Compliance Score", f"{score:.1f}%")
    m2.metric("Fully Met", counts.get("MET", 0))
    m3.metric("Partially Met", counts.get("PARTIALLY_MET", 0))
    m4.metric("Not Met", counts.get("NOT_MET", 0))
    m5.metric("Conflicting", counts.get("CONFLICTING", 0))
    m6.metric("Risk Posture", risk_level)

    st.markdown("### 📥 Export Compliance Reports")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        pdf_url = f"{BACKEND_URL}/api/analyze/{st.session_state.job_id}/download"
        try:
            resp = requests.get(pdf_url)
            if resp.status_code == 200 and resp.content.startswith(b"%PDF-"):
                st.download_button(
                    label="📄 Download Official PDF Compliance Report",
                    data=resp.content,
                    file_name=f"{company_name}_GDPR_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.button("📄 PDF Report Downloading...", disabled=True)
        except Exception:
            st.button("📄 PDF Report Downloading...", disabled=True)

    with d_col2:
        json_str = json.dumps(res, indent=2)
        st.download_button(
            label="💾 Download Full Analysis JSON",
            data=json_str,
            file_name=f"{company_name}_GDPR_Analysis.json",
            mime="application/json",
            use_container_width=True,
        )

    # Key Gaps & Remediation Section
    if res.get("key_gaps"):
        st.markdown("### ⚠️ Key Compliance Gaps")
        for gap in res["key_gaps"]:
            st.warning(f"**Group {gap.get('group_id')} (Article {gap.get('article_number')})**: {gap.get('gap')}")

    # Detailed Findings Section
    st.markdown("### 🔍 Requirement Group Findings")
    for grp in all_groups:
        with st.expander(f"Group {grp.get('group_id','')} — {grp.get('principle','')} | Status: {grp.get('status','')}"):
            st.write(f"**Condition Logic:** `{grp.get('condition_logic','')}` | **Confidence:** `{grp.get('confidence', 0.0)*100:.0f}%`")
            if grp.get("reason"):
                st.write(f"**Summary:** {grp['reason']}")
            if grp.get("gap"):
                st.warning(f"**Identified Gap:** {grp['gap']}")

            st.markdown("#### Sub-Obligations Breakdown:")
            for ob in grp.get("sub_obligations", []):
                st.markdown(f"- **{ob.get('obligation_id','')}**: Status `{ob.get('status','')}`")
                st.caption(f"Reason: {ob.get('reason','')}")
                for ev in ob.get("evidence", []):
                    st.info(f"Quote ({ev.get('chunk_id','')}): \"{ev.get('quote','')}\"")
