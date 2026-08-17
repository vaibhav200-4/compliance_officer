# test_report_agent_unit.py

from app.agents.report_agent import ReportAgent

def test_report_agent_in_memory():
    agent = ReportAgent(output_dir="Data/test_output")

    # 1. Consume group 14.1
    group_14_1 = {
        "article_number": 14,
        "group_id": "14.1",
        "principle": "Controller identification",
        "status": "MET",
        "sub_obligations": []
    }
    agent.consume_result(group_14_1)

    # 2. Consume group 14.2
    group_14_2 = {
        "article_number": 14,
        "group_id": "14.2",
        "principle": "Purpose of processing",
        "status": "NOT_MET",
        "sub_obligations": []
    }
    agent.consume_result(group_14_2)

    # 3. Consume group 14.1 again (duplicate)
    agent.consume_result(group_14_1)

    # 4. Consume article 14 summary
    article_14_summary = {
        "article_number": 14,
        "article_title": "Information to be provided where personal data have not been obtained from data subject",
        "status": "PARTIALLY_MET",
        "confidence": 0.75,
        "groups": [group_14_1, group_14_2]
    }
    agent.consume_result(article_14_summary)

    # Verify progress
    progress = agent.get_progress()
    print("Progress:", progress)
    assert progress["articles_received"] == 1
    assert progress["groups_received"] == 2

    # Finalize report in memory
    report = agent.finalize_report(company_name="Test Co", policy_name="Test Policy")
    
    assert report["company_name"] == "Test Co"
    assert len(report["all_groups"]) == 2
    
    group_ids = [g["group_id"] for g in report["all_groups"]]
    assert sorted(group_ids) == ["14.1", "14.2"]

    print("UNIT TEST PASSED CLEANLY!")

if __name__ == "__main__":
    test_report_agent_in_memory()
