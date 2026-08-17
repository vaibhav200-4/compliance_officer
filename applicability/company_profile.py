from pydantic import BaseModel
from report.llm_called.llm_client import get_gemini_llm

class CompanyProfile(BaseModel):
    international_transfer: bool
    processes_children_data: bool
    processes_special_category: bool
    automated_decision_making: bool
    large_scale_monitoring: bool
    joint_controllers: bool
    uses_processors: bool
    data_breach_occurred: bool
    public_authority: bool
    profiling: bool

PROFILE_PROMPT = """Read this privacy policy and determine which of these apply to the company,
based only on what's stated or clearly implied. Default to False if unclear/not mentioned.

Policy text:
{policy_text}

Return ONLY JSON matching the CompanyProfile schema (all boolean fields).
"""

def extract_company_profile(full_policy_text: str) -> CompanyProfile:
    #llm = get_llm(model="llama-3.3-70b-versatile")  # main model, one call only
    llm = get_gemini_llm(model="gemini-3.1-flash-lite")
    return llm.with_structured_output(CompanyProfile).invoke(
        PROFILE_PROMPT.format(policy_text=full_policy_text[:12000])  # cap tokens
    )

   #gemini-3.1-flash-lite