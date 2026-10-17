"""QA answer metrics, following Search-R1 / SQuAD normalization."""

import re
import string


def normalize_answer(s: str) -> str:
    s = str(s).lower()  # coerce: some CRAG golds are ints (e.g. a year)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred: str | None, golden: list[str]) -> int:
    if pred is None:
        return 0
    p = normalize_answer(pred)
    return int(any(p == normalize_answer(g) for g in golden))


def cover_em(pred: str | None, golden: list[str]) -> int:
    """Golden answer contained in prediction (sub-EM)."""
    if pred is None:
        return 0
    p = normalize_answer(pred)
    return int(any(normalize_answer(g) in p for g in golden))


def _f1(pred: str, gold: str) -> float:
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = {}
    for t in pt:
        if t in gt:
            common[t] = min(pt.count(t), gt.count(t))
    nsame = sum(common.values())
    if nsame == 0:
        return 0.0
    prec, rec = nsame / len(pt), nsame / len(gt)
    return 2 * prec * rec / (prec + rec)


def f1(pred: str | None, golden: list[str]) -> float:
    """Max token-level F1 over gold answers (SQuAD/HotpotQA style)."""
    if not pred:
        return 0.0
    return max(_f1(pred, g) for g in golden)
