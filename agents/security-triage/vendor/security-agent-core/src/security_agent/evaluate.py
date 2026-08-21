"""Classification metrics: binary vulnerability detection + multi-class CWE.

`task_result` extracts one prediction's per-task outcome (binary label + CWE);
`aggregate` turns a list of them into two separate, standard classification reports:

  * ``binary`` — vulnerable-vs-benign: accuracy / precision / recall / F1 / confusion,
    plus the **paired** accuracy that defeats the "always say vulnerable" policy.
  * ``cwe`` — multi-class CWE over all vulnerable functions: accuracy + macro / weighted
    precision / recall / F1 (with a per-class breakdown).
"""
from __future__ import annotations

from typing import Optional

from .verdict import normalize_cwe


def task_result(pred: dict, gold: dict) -> dict:
    """Extract one prediction's per-task classification outcome.

    `gold` needs `gold_is_vulnerable` and `gold_cwe`; `pred` is a parsed verdict from
    `verdict.parse_verdict`. A CWE is recorded only when the model calls the function
    vulnerable (`pred_cwe` is None otherwise); `gold_cwe` is None for benign functions.
    """
    gold_vuln = bool(gold.get("gold_is_vulnerable"))
    gold_cwe = normalize_cwe(gold.get("gold_cwe")) if gold_vuln else None

    if pred is None or pred.get("parse_error"):
        return {
            "parse_error": True,
            "gold_is_vulnerable": gold_vuln,
            "gold_cwe": gold_cwe,
            "pred_is_vulnerable": None,
            "pred_cwe": None,
        }

    pred_vuln = bool(pred.get("is_vulnerable"))
    return {
        "parse_error": False,
        "gold_is_vulnerable": gold_vuln,
        "gold_cwe": gold_cwe,
        "pred_is_vulnerable": pred_vuln,
        # A CWE only means something when the model flagged the function vulnerable.
        "pred_cwe": normalize_cwe(pred.get("cwe")) if pred_vuln else None,
    }


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 from a single class's tp/fp/fn counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _binary_label(scored: dict) -> bool:
    """The predicted binary label, coercing an unparseable verdict to the wrong label."""
    gold_v = bool(scored["gold_is_vulnerable"])
    pred_v = scored["pred_is_vulnerable"]
    return (not gold_v) if pred_v is None else pred_v


def _binary_report(results: list[dict]) -> dict:
    """Binary vulnerable-vs-benign metrics over every task, plus paired accuracy."""
    n = len(results)
    tp = fp = fn = tn = 0
    label_correct = 0
    for r in results:
        gold_v = bool(r["scored"]["gold_is_vulnerable"])
        pred_v = _binary_label(r["scored"])
        label_correct += int(pred_v == gold_v)
        if gold_v and pred_v:
            tp += 1
        elif not gold_v and pred_v:
            fp += 1
        elif gold_v and not pred_v:
            fn += 1
        else:
            tn += 1
    precision, recall, f1 = _prf(tp, fp, fn)

    # Paired accuracy: a pair counts only if both members' labels are correct.
    by_pair: dict[str, list[dict]] = {}
    for r in results:
        by_pair.setdefault(r["pair_id"], []).append(r)
    full_pairs = [p for p in by_pair.values() if len(p) >= 2]
    paired_correct = sum(
        1 for p in full_pairs
        if all(_binary_label(m["scored"]) == bool(m["scored"]["gold_is_vulnerable"]) for m in p)
    )
    paired_accuracy = paired_correct / len(full_pairs) if full_pairs else None

    return {
        "n": n,
        "accuracy": round(label_correct / n, 4) if n else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "paired_accuracy": None if paired_accuracy is None else round(paired_accuracy, 4),
        "n_pairs": len(full_pairs),
    }


def _cwe_report(results: list[dict]) -> Optional[dict]:
    """Multi-class CWE metrics over ALL truly-vulnerable functions (fixed population).

    y_true is the gold CWE; y_pred is the model's CWE, or None when it called the
    function benign — a missed detection counts as a CWE error. Macro/weighted averages
    are taken over the true CWE classes; a None prediction never matches, so it depresses
    the recall of the corresponding class. Returns None if the set has no vulnerable
    functions.
    """
    y = [
        (r["scored"]["gold_cwe"], r["scored"]["pred_cwe"])
        for r in results if r["scored"]["gold_is_vulnerable"]
    ]
    n = len(y)
    if n == 0:
        return None
    correct = sum(1 for t, p in y if p == t)
    classes = sorted({t for t, _ in y})

    per_class: dict[str, dict] = {}
    macro_p = macro_r = macro_f = 0.0
    w_p = w_r = w_f = 0.0
    for c in classes:
        tp = sum(1 for t, p in y if t == c and p == c)
        fp = sum(1 for t, p in y if t != c and p == c)
        fn = sum(1 for t, p in y if t == c and p != c)
        support = sum(1 for t, _ in y if t == c)
        p_, r_, f_ = _prf(tp, fp, fn)
        per_class[c] = {
            "precision": round(p_, 4), "recall": round(r_, 4),
            "f1": round(f_, 4), "support": support,
        }
        macro_p += p_; macro_r += r_; macro_f += f_
        w_p += p_ * support; w_r += r_ * support; w_f += f_ * support

    k = len(classes)
    return {
        "n": n,
        "num_classes": k,
        "accuracy": round(correct / n, 4),
        "macro": {
            "precision": round(macro_p / k, 4),
            "recall": round(macro_r / k, 4),
            "f1": round(macro_f / k, 4),
        },
        "weighted": {
            "precision": round(w_p / n, 4),
            "recall": round(w_r / n, 4),
            "f1": round(w_f / n, 4),
        },
        "per_class": per_class,
    }


def aggregate(results: list[dict]) -> dict:
    """Aggregate per-task results into separate binary and CWE classification reports.

    Each item is `{"pair_id": str, "scored": <task_result output>}`.
    """
    n = len(results)
    if n == 0:
        return {"n": 0, "binary": None, "cwe": None}
    return {"n": n, "binary": _binary_report(results), "cwe": _cwe_report(results)}
