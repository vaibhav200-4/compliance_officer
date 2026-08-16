"""
Compliance Guardian -- Streamlit UI
------------------------------------
Talks to the FastAPI backend (api/main.py) running separately on
http://127.0.0.1:8000.

Run (from wherever you place this file):
    streamlit run app.py

Make sure the FastAPI backend is ALSO running in another terminal:
    python -m uvicorn api.main:app --reload   (from compliance_officer/ root)
"""

import base64
import time

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Compliance Guardian", layout="wide", page_icon="🛡️")

# ============================================================
# THEME / CSS -- dark navy + neon blue, matching the mockup
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 20% 20%, #0d1b2e 0%, #060a12 60%, #030509 100%);
        color: #e5edf7;
    }
    #MainMenu, footer, header {visibility: hidden;}

    .cg-navbar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 14px 6px 20px 6px; border-bottom: 1px solid rgba(56,189,248,0.15);
        margin-bottom: 30px;
    }
    .cg-logo { font-size: 22px; font-weight: 800; letter-spacing: 1px; color: #38bdf8;
               text-shadow: 0 0 12px rgba(56,189,248,0.6); }
    .cg-logo span { display: block; font-size: 14px; font-weight: 400; color: #e5edf7; text-shadow: none; }
    .cg-navlinks { display: flex; gap: 28px; color: #9fb3c8; font-size: 14px; }
    .cg-navlinks .active { color: #38bdf8; border-bottom: 2px solid #38bdf8; padding-bottom: 4px; }

    .cg-title {
        text-align: center; font-size: 52px; font-weight: 900; letter-spacing: 4px;
        color: #38bdf8; text-shadow: 0 0 30px rgba(56,189,248,0.55);
        margin: 10px 0 36px 0;
    }

    .cg-card {
        background: linear-gradient(180deg, rgba(19,33,54,0.9) 0%, rgba(10,18,30,0.9) 100%);
        border: 1px solid rgba(56,189,248,0.25);
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 22px;
        box-shadow: 0 0 24px rgba(56,189,248,0.06);
    }
    .cg-card-label { color: #7fa8c9; font-size: 12px; letter-spacing: 1.5px;
                      text-transform: uppercase; margin-bottom: 10px; }
    .cg-row { display: flex; align-items: center; gap: 14px; }
    .cg-pdf-icon { font-size: 30px; }
    .cg-filename { font-size: 16px; font-weight: 600; color: #e5edf7; }
    .cg-status { font-size: 13px; color: #9fb3c8; margin-top: 2px; }
    .cg-check { color: #34d399; font-size: 20px; }
    .cg-spinner-text { color: #38bdf8; }

    div.stButton > button {
        background: linear-gradient(135deg, #2563eb, #38bdf8);
        color: white; border: none; border-radius: 30px;
        padding: 10px 30px; font-weight: 700; letter-spacing: 0.5px;
        box-shadow: 0 0 20px rgba(56,189,248,0.4);
    }
    div.stButton > button:hover { box-shadow: 0 0 30px rgba(56,189,248,0.7); }

    section[data-testid="stFileUploaderDropzone"] {
        background: rgba(15,26,43,0.6) !important;
        border: 2px dashed rgba(56,189,248,0.4) !important;
        border-radius: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# NAVBAR (decorative, matches the mockup)
# ============================================================
st.markdown(
    """
    <div class="cg-navbar">
        <div class="cg-logo">COMPLIANCE<span>GUARDIAN</span></div>
        <div class="cg-navlinks">
            <span class="active">Dashboard</span><span>Policies</span>
            <span>Analytics</span><span>Reports</span><span>Settings</span>
        </div>
    </div>
    <div class="cg-title">COMPLIANCE OFFICER</div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "job_id": None,
    "status": "idle",       # idle | uploading | running | done | failed
    "stage": None,
    "message": None,
    "filename": None,
    "report_bytes": None,
    "report_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_job():
    for k, v in defaults.items():
        st.session_state[k] = v


# ============================================================
# UPLOAD FORM (only shown when idle / not currently running a job)
# ============================================================
if st.session_state.status in ("idle", "done", "failed"):
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your policy (PDF)", type=["pdf"], label_visibility="collapsed"
        )
    with col2:
        company = st.text_input("Company name", placeholder="e.g. Acme Corp")

    if st.button("➜  SEND FOR REVIEW", use_container_width=False):
        if not uploaded_file:
            st.warning("Pehle ek PDF upload karo.")
        elif not company.strip():
            st.warning("Company name daalo.")
        else:
            reset_job()
            st.session_state.filename = uploaded_file.name
            st.session_state.status = "uploading"
            try:
                resp = requests.post(
                    f"{API_BASE}/upload-policy",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    data={"company": company.strip(), "policy_version": "v1.0", "skip_ingest": "false"},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.job_id = data["job_id"]
                st.session_state.status = "running"
                st.session_state.message = "Pipeline started"
            except Exception as e:
                st.session_state.status = "failed"
                st.session_state.message = f"Upload failed: {e}"
            st.rerun()

# ============================================================
# INPUT PDF CARD (shown once a job exists)
# ============================================================
if st.session_state.filename:
    if st.session_state.status == "uploading":
        status_html = '<span class="cg-spinner-text">⏳ Uploading...</span>'
    elif st.session_state.status == "running":
        status_html = f'<span class="cg-check">✅</span> Uploaded &nbsp; · &nbsp; <span class="cg-spinner-text">⚙ {st.session_state.stage or "starting"} — {st.session_state.message or ""}</span>'
    elif st.session_state.status == "done":
        status_html = '<span class="cg-check">✅</span> Uploaded &nbsp; · &nbsp; Analysis complete'
    elif st.session_state.status == "failed":
        status_html = f'<span class="cg-check">✅</span> Uploaded &nbsp; · &nbsp; ❌ {st.session_state.message or "Failed"}'
    else:
        status_html = ""

    st.markdown(
        f"""
        <div class="cg-card">
            <div class="cg-card-label">Input · Privacy Policy</div>
            <div class="cg-row">
                <div class="cg-pdf-icon">📄</div>
                <div>
                    <div class="cg-filename">{st.session_state.filename}</div>
                    <div class="cg-status">{status_html}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# POLL STATUS while running
# ============================================================
if st.session_state.status == "running" and st.session_state.job_id:
    try:
        resp = requests.get(f"{API_BASE}/status/{st.session_state.job_id}", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        st.session_state.stage = data.get("stage")
        st.session_state.message = data.get("message")
        if data.get("status") == "done":
            st.session_state.status = "done"
        elif data.get("status") == "failed":
            st.session_state.status = "failed"
            st.session_state.message = data.get("error") or data.get("message")
    except Exception as e:
        st.session_state.message = f"Status check failed: {e}"

    if st.session_state.status == "running":
        time.sleep(3)
        st.rerun()
    else:
        st.rerun()

# ============================================================
# OUTPUT REPORT CARD (shown once done)
# ============================================================
if st.session_state.status == "done":
    if st.session_state.report_bytes is None:
        try:
            resp = requests.get(f"{API_BASE}/download-report/{st.session_state.job_id}", timeout=60)
            resp.raise_for_status()
            st.session_state.report_bytes = resp.content
            cd = resp.headers.get("content-disposition", "")
            st.session_state.report_name = cd.split("filename=")[-1].strip('"') if "filename=" in cd else "compliance_report.pdf"
        except Exception as e:
            st.error(f"Report download failed: {e}")

    if st.session_state.report_bytes:
        st.markdown(
            f"""
            <div class="cg-card">
                <div class="cg-card-label">Output · Compliance Report</div>
                <div class="cg-row">
                    <div class="cg-pdf-icon">📊</div>
                    <div>
                        <div class="cg-filename">{st.session_state.report_name}</div>
                        <div class="cg-status"><span class="cg-check">✅</span> Ready</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b64 = base64.b64encode(st.session_state.report_bytes).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="700" '
            f'style="border-radius:12px;border:1px solid rgba(56,189,248,0.25);"></iframe>',
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download Report",
            data=st.session_state.report_bytes,
            file_name=st.session_state.report_name,
            mime="application/pdf",
        )

        if st.button("Start new review"):
            reset_job()
            st.rerun()

# ============================================================
# FAILED STATE -- allow retry
# ============================================================
if st.session_state.status == "failed":
    st.error(st.session_state.message or "Something went wrong.")
    if st.button("Try again"):
        reset_job()
        st.rerun()