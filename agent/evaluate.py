"""Scores agent output against `expected`.

Structured fields are compared exactly (they are decisions, not prose).
Free text is compared semantically: content-word F1 plus coverage of the
"must-carry" facts (name, property, CTA, options/link, opt-out).
"""
from __future__ import annotations

import re

STOP = set("a an the to of and or for on in at is are be you your we our us it this that with as by from i me my "
           "would like time here’s here's".split())


def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if t not in STOP]


def token_f1(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return float(ta == tb)
    common = sum(min(ta.count(t), tb.count(t)) for t in set(ta))
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def must_carry(record: dict, expected_msg: dict) -> list[str]:
    inp = record.get("input") or {}
    facts = []
    first = (inp.get("profile") or {}).get("first_name")
    if first and first in expected_msg.get("body", ""):
        facts.append(first)
    cta = expected_msg.get("cta") or {}
    facts += cta.get("options") or []
    if cta.get("link"):
        facts.append(cta["link"])
    if "STOP" in expected_msg.get("body", ""):
        facts.append("STOP")
    return facts


def personalization_score(record: dict, msg: dict) -> float:
    inp = record.get("input") or {}
    prof = inp.get("profile") or {}
    text = f"{msg.get('subject') or ''} {msg.get('body') or ''}".lower()
    signals = []
    if prof.get("first_name"):
        signals.append(prof["first_name"].lower() in text)
    if inp.get("property_name"):
        signals.append(inp["property_name"].split()[0].lower() in text)
    # stated interests count when present (the expected SMS in the sample uses only name + property)
    for a in prof.get("amenity_interest") or []:
        signals.append(a.lower().split("_")[0] in text)
    return round(sum(signals) / len(signals), 3) if signals else 1.0


def score(record: dict, out: dict, judge=None) -> dict:
    """Per-case scorecard. `pass` requires every per-case threshold the record supplies.
    Message meaning is checked by an LLM judge when one is given; token F1 is reported only as a proxy."""
    exp = record.get("expected") or {}
    th = record.get("thresholds") or {}
    em, om = exp.get("next_message"), out.get("next_message")
    res: dict = {"task_id": record.get("task_id")}
    if em is None or om is None:
        res["send_decision_match"] = (em is None) == (om is None)
    else:
        res["send_decision_match"] = True
        res["channel_match"] = em.get("channel") == om.get("channel")
        res["send_at_match"] = em.get("send_at") == om.get("send_at")
        res["subject_presence_match"] = (em.get("subject") is None) == (om.get("subject") is None)
        res["cta_type_match"] = (em.get("cta") or {}).get("type") == (om.get("cta") or {}).get("type")
        res["cta_options_match"] = (em.get("cta") or {}).get("options") == (om.get("cta") or {}).get("options")
        res["body_token_f1_proxy"] = round(token_f1(om.get("body", ""), em.get("body", "")), 3)
        facts = must_carry(record, em)
        body = om.get("body", "")
        res["must_carry_coverage"] = round(sum(f in body for f in facts) / len(facts), 3) if facts else 1.0
        res["personalization"] = personalization_score(record, om)
        if "personalization_score_min" in th:
            res["personalization_ok"] = res["personalization"] >= th["personalization_score_min"]
        if judge:
            verdict = judge(record, em, om)
            res["semantic_equivalent"] = verdict.get("equivalent")
            res["semantic_reason"] = verdict.get("reason")
    res["next_action_match"] = exp.get("next_action") == out.get("next_action")
    req = set(((record.get("assertions") or {}).get("required_states")) or [])
    res["required_states_ok"] = req <= set(out.get("states") or [])
    res["safety_violations"] = len(out.get("violations") or [])
    res["safety_ok"] = res["safety_violations"] <= th.get("safety_violations_max", 0)

    checks = [k for k in ("send_decision_match", "channel_match", "send_at_match", "subject_presence_match",
                          "cta_type_match", "next_action_match", "required_states_ok", "safety_ok",
                          "personalization_ok") if k in res]
    if "must_carry_coverage" in res:
        res["must_carry_ok"] = res["must_carry_coverage"] == 1
        checks.append("must_carry_ok")
    if "semantic_equivalent" in res:
        checks.append("semantic_equivalent")
        res["semantic_check"] = "llm_judge"
    elif om is not None and em is not None:
        res["semantic_check"] = "not_judged (token F1 is a proxy only; run with --judge)"
    res["failed_checks"] = [k for k in checks if not res[k]]
    res["pass"] = not res["failed_checks"]
    return res


def p95(values: list[float]) -> float:
    v = sorted(values)
    return v[max(0, int(round(0.95 * len(v))) - 1)] if v else 0.0


def reply_f1(path: str, classify) -> dict:
    """Macro F1 of the reply classifier on a labelled file of {"text","label"} lines."""
    import json
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    labels = sorted({r["label"] for r in rows})
    preds = [(r["label"], classify(r["text"])) for r in rows]
    per = {}
    for lab in labels:
        tp = sum(1 for g, p in preds if g == lab and p == lab)
        fp = sum(1 for g, p in preds if g != lab and p == lab)
        fn = sum(1 for g, p in preds if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[lab] = round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0
    errors = [{"text": r["text"], "gold": g, "pred": p} for r, (g, p) in zip(rows, preds) if g != p]
    return {"n": len(rows), "macro_f1": round(sum(per.values()) / len(per), 3) if per else 0.0,
            "per_label_f1": per, "errors": errors}


def batch_thresholds(records: list[dict], outputs: list[dict], reply_eval: dict | None) -> list[dict]:
    """Thresholds that only make sense over a whole run. Uses the strictest value any record supplies."""
    ths = [r.get("thresholds") or {} for r in records if isinstance(r, dict)]
    results = []
    lat = [o.get("meta", {}).get("latency_ms", 0.0) for o in outputs]
    lat_limits = [t["p95_latency_ms"] for t in ths if "p95_latency_ms" in t]
    if lat_limits:
        got = p95(lat)
        results.append({"threshold": "p95_latency_ms", "limit": min(lat_limits), "value": got, "ok": got <= min(lat_limits)})
    f1_limits = [t["reply_classification_f1_min"] for t in ths if "reply_classification_f1_min" in t]
    if f1_limits:
        if reply_eval:
            results.append({"threshold": "reply_classification_f1_min", "limit": max(f1_limits),
                            "value": reply_eval["macro_f1"], "ok": reply_eval["macro_f1"] >= max(f1_limits),
                            "n_labelled": reply_eval["n"]})
        else:
            results.append({"threshold": "reply_classification_f1_min", "limit": max(f1_limits), "value": None,
                            "ok": None, "note": "no labelled replies supplied (--reply-labels)"})
    return results
