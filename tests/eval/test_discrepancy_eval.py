"""Deterministic evaluation of the hybrid discrepancy detector.

Scores precision / recall / F1 over the gold set in gold_discrepancy_set.py.
Pure code — no API keys — so it runs in CI and yields real numbers you can cite
for the product's differentiator (Docs+DB discrepancy detection).

Run:  pytest tests/eval/test_discrepancy_eval.py -s
"""

from api.utils.discrepancy_detector import detect_discrepancies
from tests.eval.gold_discrepancy_set import GOLD_CASES


def _tokens(label: str) -> set[str]:
    return {t for t in label.lower().replace("_", " ").split() if t}


def _matches(expected_label: str, detected_labels: set[str]) -> bool:
    """True if some detected label covers all tokens of the expected label."""
    want = _tokens(expected_label)
    return any(want <= _tokens(d) for d in detected_labels)


def _score():
    tp = fp = fn = 0
    per_case = []
    for case in GOLD_CASES:
        report = detect_discrepancies(case.doc_context, case.sql_metadata)
        detected = {d.label for d in report.discrepancies}

        matched_expected = {e for e in case.expected if _matches(e, detected)}
        # A detected label is a true positive if it covers an expected metric.
        matched_detected = {
            d for d in detected
            if any(_tokens(e) <= _tokens(d) for e in case.expected)
        }

        c_tp = len(matched_expected)
        c_fn = len(case.expected) - c_tp
        c_fp = len(detected) - len(matched_detected)

        tp += c_tp
        fp += c_fp
        fn += c_fn
        per_case.append((case.name, c_tp, c_fp, c_fn, sorted(detected), sorted(case.expected)))

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn, per_case


def test_discrepancy_precision_recall(capsys):
    precision, recall, f1, tp, fp, fn, per_case = _score()

    with capsys.disabled():
        print("\n=== Discrepancy Detector Evaluation ===")
        print(f"cases={len(GOLD_CASES)}  TP={tp}  FP={fp}  FN={fn}")
        print(f"precision={precision:.2f}  recall={recall:.2f}  F1={f1:.2f}\n")
        for name, c_tp, c_fp, c_fn, detected, expected in per_case:
            flag = "OK " if (c_fp == 0 and c_fn == 0) else "!! "
            print(f"  {flag}{name}: TP={c_tp} FP={c_fp} FN={c_fn} "
                  f"detected={detected} expected={expected}")

    # Precision is the demo-critical metric — false positives (the -100% /
    # +19566% garbage) destroy trust. Guard it hard. Recall floor is lenient:
    # some row-shape cases (segment value labeled by column) are known gaps.
    assert precision >= 0.85, f"precision regressed to {precision:.2f}"
    assert recall >= 0.5, f"recall regressed to {recall:.2f}"
    assert fp == 0, f"{fp} false-positive discrepancies (must be 0 for demo trust)"
