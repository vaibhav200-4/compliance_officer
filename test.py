from app.compliance.gdpr_kb import GDPRKnowledgeBase


def main():
    print("=" * 60)
    print("GDPR KNOWLEDGE BASE TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load knowledge base
    # ---------------------------------------------------------

    print("\n[1] Loading GDPR knowledge base...")

    kb = GDPRKnowledgeBase()

    print("SUCCESS")
    print(f"Articles       : {kb.article_count}")
    print(f"Sub-obligations: {kb.obligation_count}")

    # ---------------------------------------------------------
    # 2. Test Article 5
    # ---------------------------------------------------------

    print("\n[2] Loading Article 5...")

    article = kb.require_article(5)

    print(f"Article number : {article.article_number}")
    print(f"Article name   : {article.article_name}")
    print(f"Checkability   : {article.checkability}")
    print(f"Obligations    : {len(article.sub_obligations)}")

    # ---------------------------------------------------------
    # 3. Print obligations
    # ---------------------------------------------------------

    print("\n[3] Article 5 obligations")
    print("-" * 60)

    for obligation in article.sub_obligations:
        print(
            f"{obligation.id:<12} "
            f"| group={obligation.parent_group_id:<5} "
            f"| condition={obligation.condition_logic}"
        )

    # ---------------------------------------------------------
    # 4. Test specific obligation
    # ---------------------------------------------------------

    print("\n[4] Testing obligation 5.1.e.1...")

    obligation = kb.require_sub_obligation("5.1.e.1")

    print(f"ID                : {obligation.id}")
    print(f"Parent group      : {obligation.parent_group_id}")
    print(f"Legal text        : {obligation.legal_text}")
    print(f"Plain summary     : {obligation.plain_summary}")
    print(f"Applicability     : {obligation.applicability_condition}")
    print(f"Evidence prompt   : {obligation.evidence_prompt}")

    # ---------------------------------------------------------
    # 5. Conditional obligations
    # ---------------------------------------------------------

    print("\n[5] Conditional obligations in Article 5")
    print("-" * 60)

    conditional = kb.get_conditional_obligations(5)

    for item in conditional:
        print(
            f"{item.id}: "
            f"{item.applicability_condition}"
        )

    # ---------------------------------------------------------
    # 6. Unconditional obligations
    # ---------------------------------------------------------

    print("\n[6] Unconditional obligations in Article 5")
    print("-" * 60)

    unconditional = kb.get_unconditional_obligations(5)

    for item in unconditional:
        print(item.id)

    # ---------------------------------------------------------
    # 7. Search test
    # ---------------------------------------------------------

    print("\n[7] Keyword search: retention")
    print("-" * 60)

    results = kb.search_obligations("retention")

    for item in results:
        print(
            f"{item.id}: {item.plain_summary}"
        )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("GDPR KNOWLEDGE BASE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()