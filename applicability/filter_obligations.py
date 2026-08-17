def filter_applicable(all_obligations: list, profile) -> tuple[list, list]:
    """Returns (applicable_obligations, not_applicable_obligations)."""
    profile_dict = profile.dict() if hasattr(profile, "dict") else profile
    applicable, not_applicable = [], []
    for obl in all_obligations:
        conditions = obl.get("applicability_conditions", [])
        if not conditions or any(profile_dict.get(c, False) for c in conditions):
            applicable.append(obl)
        else:
            not_applicable.append(obl)
    return applicable, not_applicable