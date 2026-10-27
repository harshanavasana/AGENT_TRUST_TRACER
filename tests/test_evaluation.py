from tracer.evaluation import (
    LabeledEvaluationCase,
    VariantOutcome,
    ablation_report,
)


def test_ablation_metrics_include_safe_but_suspicious_false_positives() -> None:
    report = ablation_report(
        [
            LabeledEvaluationCase(
                expected_violation=True,
                safe_trace=False,
                expected_origin_step=1,
                expected_manifestation_step=3,
                outcomes={
                    "rules_only": VariantOutcome(
                        verdict="PASS",
                        action="CONTINUE",
                        detected=False,
                    ),
                    "rules_plus_judge": VariantOutcome(
                        verdict="FAIL",
                        action="BLOCK",
                        detected=True,
                        origin_step=1,
                        manifestation_step=3,
                        latency_ms=24,
                        input_tokens=200,
                        output_tokens=50,
                    ),
                },
            ),
            LabeledEvaluationCase(
                expected_violation=False,
                safe_trace=True,
                outcomes={
                    "rules_only": VariantOutcome(
                        verdict="PASS",
                        action="CONTINUE",
                        detected=False,
                    ),
                    "rules_plus_judge": VariantOutcome(
                        verdict="REVIEW",
                        action="REVIEW",
                        detected=False,
                    ),
                },
            ),
        ]
    )

    assert report["status"] == "ready"
    assert report["variants"]["rules_plus_judge"]["detection_rate"] == 1
    assert report["variants"]["rules_plus_judge"]["origin_accuracy"] == 1
    assert report["variants"]["rules_plus_judge"]["false_positives"] == 1
    assert report["variants"]["rules_plus_judge"]["safe_but_suspicious"] == 1
    assert report["variants"]["rules_plus_judge"]["input_tokens"] == 200


def test_ablation_does_not_claim_scores_without_labeled_cases() -> None:
    report = ablation_report([])

    assert report["status"] == "no_labeled_cases"
    assert report["case_count"] == 0
    assert report["variants"]["rules_only"]["detection_rate"] is None
