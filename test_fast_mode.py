# test_fast_mode.py

import os
import sys
from pathlib import Path

# Enable FAST_TEST_MODE
os.environ["FAST_TEST_MODE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

# Force backend path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/app.log", level="INFO")

from app.agents.orchestrator import ComplianceOrchestrator
from app.compliance.gdpr_kb import GDPRKnowledgeBase

def main():
    print("===========================================================================")
    print("FAST TEST MODE VERIFICATION: Articles 14 and 15")
    print("===========================================================================")

    kb = GDPRKnowledgeBase()
    orchestrator = ComplianceOrchestrator(
        knowledge_base=kb,
        max_workers=2,
        top_k=5,
        output_dir="Data/analysis_results",
    )

    result = orchestrator.run(
        start_article=14,
        end_article=15,
        resume=False,
    )

    print("===========================================================================")
    print(f"FAST TEST RESULT STATUS: {result.get('status')}")
    print(f"COMPLETED ARTICLES    : {result.get('completed')}/{result.get('total')}")
    print(f"FINAL REPORT READY    : {result.get('final_report') is not None}")
    print("===========================================================================")

if __name__ == "__main__":
    main()
