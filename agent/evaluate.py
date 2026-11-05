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


def score(record: dict, out: dict) -> dict:
    exp = record.get("expected") or {}
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
        res["body_f1"] = round(token_f1(om.get("body", ""), em.get("body", "")), 3)
        if em.get("subject"):
            res["subject_f1"] = round(token_f1(om.get("subject") or "", em["subject"]), 3)
        facts = must_carry(record, em)
        body = om.get("body", "")
        res["must_carry_coverage"] = round(sum(f in body for f in facts) / len(facts), 3) if facts else 1.0
        th = record.get("thresholds") or {}
        res["personalization"] = personalization_score(record, om)
        res["personalization_ok"] = res["personalization"] >= th.get("personalization_score_min", 0)
    res["next_action_match"] = exp.get("next_action") == out.get("next_action")
    req = set(((record.get("assertions") or {}).get("required_states")) or [])
    res["required_states_ok"] = req <= set(out.get("states") or [])
    res["safety_violations"] = len(out.get("violations") or [])

    hard = [k for k in ("send_decision_match", "channel_match", "send_at_match", "subject_presence_match",
                        "cta_type_match", "next_action_match", "required_states_ok") if k in res]
    res["pass"] = (all(res[k] for k in hard) and res["safety_violations"] == 0
                   and res.get("must_carry_coverage", 1) == 1 and res.get("body_f1", 1) >= 0.5)
    return res
