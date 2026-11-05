"""Run: .venv/bin/python -m pytest -q  (or python tests/test_agent.py)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import guardrails  # noqa: E402
from agent.agent import MessagingAgent, classify_reply  # noqa: E402
from agent.policy import decide  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
SAMPLE = [json.loads(l) for l in (DATA / "sample.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def test_samples_match_expected_structure():
    agent = MessagingAgent(examples=SAMPLE)
    for r in SAMPLE:
        out = agent.run(r)
        exp = r["expected"]
        assert out["next_message"]["channel"] == exp["next_message"]["channel"]
        assert out["next_message"]["send_at"] == exp["next_message"]["send_at"]
        assert out["next_action"] == exp["next_action"]
        assert set(r["assertions"]["required_states"]) <= set(out["states"])


def test_guardrails_catch_violations():
    r = SAMPLE[0]
    d = decide(r)
    bad = {"channel": "voice", "send_at": d.send_at, "subject": "x",
           "body": "Hi Taylor - perfect for families with kids! Call 214-555-0199 or email t@x.com", "cta": {}}
    _, v = guardrails.check(r, d, bad)
    text = " ".join(v)
    for needle in ("lacks consent", "fair housing", "pii", "opt-out", "subject", "CTA"):
        assert needle in text, (needle, v)


class FakeLLM:
    """First draft breaks hard limits (voice has no consent, 2am send, bad action type, steering copy);
    second draft is a valid email choice. Exercises per-field acceptance + repair."""
    def __init__(self):
        self.calls = 0

    def compose(self, record, d, shots, feedback=None):
        self.calls += 1
        if self.calls == 1:
            return {"next_message": {"channel": "voice", "send_at": "2025-12-09T02:00:00-06:00", "subject": None,
                                     "body": "Great for families! See you soon.", "cta": {"type": "x"}},
                    "next_action": {"type": "spam_them"}, "rationale": "bad"}
        return {"next_message": {"channel": "email", "send_at": "2025-12-10T11:30:00-06:00", "subject": "Tour Oak Ridge",
                                 "body": "Hi Taylor,\nBook a tour this week → https://oakridge.example/tour\nTo opt out of emails, click here or reply STOP.",
                                 "cta": {"type": "x"}},
                "next_action": {"type": "follow_up_in_days", "value": 2}, "rationale": "good"}


def test_llm_decisions_accepted_within_hard_limits():
    agent = MessagingAgent(examples=SAMPLE)
    agent.llm = FakeLLM()
    out = agent.run(SAMPLE[0])
    m = out["next_message"]
    assert agent.llm.calls == 2
    assert m["channel"] == "email" and m["send_at"] == "2025-12-10T11:30:00-06:00"   # LLM's consented choice
    assert m["cta"] == {"type": "schedule_tour", "link": "https://oakridge.example/tour"}
    assert out["next_action"] == {"type": "follow_up_in_days", "value": 2}
    assert out["meta"]["decided_by"] == {"channel": "llm", "send_at": "llm", "next_action": "llm"}
    trace = " ".join(out["meta"]["trace"])
    assert "lacks consent" in trace and "outside quiet hours" in trace and "spam_them" in trace
    assert out["violations"] == []


def test_fail_safe_when_everything_fails(monkeypatch=None):
    import agent.agent as A
    original = A.compose_template
    A.compose_template = lambda r, d: {"channel": d.channel, "send_at": d.send_at, "subject": None,
                                       "body": "Perfect for families!", "cta": {}}
    try:
        out = MessagingAgent(examples=SAMPLE).run(SAMPLE[0])
    finally:
        A.compose_template = original
    assert out["next_message"] is None
    assert out["next_action"] == {"type": "manual_review", "reason": "failed_validation"}
    assert out["meta"]["source"] == "blocked" and out["meta"]["blocked_violations"]


def test_guardrails_catch_copied_example_details():
    r = json.loads(json.dumps(SAMPLE[0]))
    r["input"]["profile"] = {}  # no name, no amenities
    d = decide(r)
    msg = {"channel": "sms", "send_at": d.send_at, "subject": None, "cta": {},
           "body": "Hi Taylor—tour our pool this week! Reply 1 for Thu, 2 for Fri. Reply STOP to opt out."}
    _, v = guardrails.check(r, d, msg)
    text = " ".join(v)
    assert "not this contact's first name" in text and "amenity 'pool'" in text, v


def test_reply_classifier():
    assert classify_reply("STOP") == "opt_out"
    assert classify_reply("2") == "select_option"
    assert classify_reply("how much is a 2br?") == "pricing_question"


def test_review_regressions():
    agent = MessagingAgent(examples=SAMPLE)
    base = {k: v for k, v in SAMPLE[0].items() if k != "expected"}

    def variant(**changes):
        r = json.loads(json.dumps(base))
        for path, val in changes.items():
            *head, last = path.split("__")
            node = r
            for h in head:
                node = node[h]
            node[last] = val
        return agent.run(r)

    # bad inputs no longer crash
    assert variant(input__timezone="CST")["next_message"]["send_at"].endswith("-06:00")
    assert variant(channel_preferences=["whatsapp", None, "SMS"])["next_message"]["channel"] == "sms"
    assert variant(input__last_interaction=1765206240)["next_message"] is not None
    # natural opt-out suppresses marketing; "stop by" does not
    assert variant(input__inbound_message="please stop texting me")["next_action"]["type"] == "opt_out"
    assert classify_reply("Can I stop by Saturday for a tour?") != "opt_out"
    # not interested closes the lead without a message
    out = variant(input__inbound_message="No thanks, found another place")
    assert out["next_message"] is None and out["next_action"]["type"] == "close_lead"
    # names that contain protected-class words are not Fair Housing violations
    out = variant(input__profile={"first_name": "Christian"}, input__property_name="White Oak Apartments")
    assert out["violations"] == [] and "fair_housing_check_passed" in out["states"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
