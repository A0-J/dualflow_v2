import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "canonical"))

from clustered_sensitivity import (  # noqa: E402
    CASES_BY_CLARITY, METHODS, direction_counts, family_case_rates, group_key,
    incidence, paired_differences, sign_test_greater, text_key, validate_complete_design,
)
from common import EXECUTOR_MODELS  # noqa: E402
from scenarios import DOMAINS  # noqa: E402


def complete_design():
    """Every family x 10 texts x its 2 cases x 4 models x 2 Planner Agent conditions; all-safe."""
    rows = []
    for domain in DOMAINS:
        for clarity, cases in CASES_BY_CLARITY.items():
            for i in range(10):
                for case in cases:
                    for model in EXECUTOR_MODELS:
                        for condition in ("context", "no_context"):
                            row = {"text_id": f"{domain}_{clarity}_{i:02d}", "domain": domain, "case": case,
                                   "executor_model": model, "planner_condition": condition,
                                   "should_execute": case != "C3"}
                            for m in METHODS:
                                row[f"{m}__unsafe"] = False
                                row[f"{m}__false_reject"] = False
                                row[f"{m}__correct"] = True
                            rows.append(row)
    return rows


def test_source_family_is_inferred_from_canonical_text_id():
    assert group_key({"text_id": "audit_ambiguous_03", "domain": "audit"}) == "audit_ambiguous"
    with pytest.raises(ValueError, match="Unexpected canonical text_id"):
        group_key({"text_id": "not-a-canonical-id", "domain": "audit"})


def test_validate_complete_design_accepts_complete_and_rejects_missing_or_duplicate():
    rows = complete_design()
    validate_complete_design(rows)
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        validate_complete_design(rows[1:])
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        validate_complete_design(rows + [dict(rows[0])])


def test_sign_test_excludes_ties_by_counting_only_signed_units():
    # 3 families higher, 0 lower, 1 tie: the tie is not part of n.
    assert sign_test_greater(3, 0) == 0.125
    assert sign_test_greater(4, 0) == 0.0625
    assert sign_test_greater(2, 2) == 0.6875
    assert sign_test_greater(0, 0) == 1.0
    with pytest.raises(ValueError, match="cannot be negative"):
        sign_test_greater(-1, 2)


def test_paired_differences_count_ties_at_both_levels():
    rows = complete_design()
    for r in rows:  # Semantic Flow only is unsafe on audit C3 only
        if r["text_id"].startswith("audit_clear") and r["case"] == "C3":
            r["semantic_only__unsafe"] = True
    families = direction_counts(paired_differences(rows, group_key), sign_test=True)
    texts = direction_counts(paired_differences(rows, text_key), sign_test=False)
    h3a_family = next(r for r in families if r["comparison"].startswith("H3a"))
    h3a_text = next(r for r in texts if r["comparison"].startswith("H3a"))
    assert (h3a_family["comparator_higher"], h3a_family["reference_higher"], h3a_family["ties"]) == (1, 0, 3)
    assert h3a_family["exact_one_sided_sign_test_p"] == 0.5
    assert (h3a_text["comparator_higher"], h3a_text["ties"]) == (10, 30)
    assert "exact_one_sided_sign_test_p" not in h3a_text  # no p-values at the text level


def test_planner_condition_comparison_pairs_conditions_within_unit():
    rows = complete_design()
    for r in rows:
        if r["text_id"] == "hr_ambiguous_04" and r["case"] == "C2" and r["planner_condition"] == "no_context":
            r["dualflow__unsafe"] = True
    texts = direction_counts(paired_differences(rows, text_key), sign_test=False)
    h5 = next(r for r in texts if r["comparison"].startswith("H5"))
    assert (h5["comparator_higher"], h5["reference_higher"], h5["units"]) == (1, 0, 40)


def test_rates_never_pool_cases_and_incidence_interval_only_for_families():
    rows = complete_design()
    rates = family_case_rates(rows)
    keys = {(r["source_family"], r["case"]) for r in rates}
    assert ("audit_clear", "C1") in keys and ("audit_clear", "C3") in keys
    assert all(r["denominator"] == 40 for r in rates if r["metric"] == "unsafe")
    text_inc = incidence(rows, text_key, with_interval=False)
    family_inc = incidence(rows, group_key, with_interval=True)
    df_text = next(r for r in text_inc if r["method"] == "dualflow" and r["metric"] == "unsafe")
    df_family = next(r for r in family_inc if r["method"] == "dualflow" and r["metric"] == "unsafe")
    assert (df_text["units"], df_text["exact_95_upper"]) == (80, "")
    assert df_family["units"] == 8 and 0.36 < df_family["exact_95_upper"] < 0.37
