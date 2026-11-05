"""Run: .venv/bin/python -m pytest -q  (or python tests/test_agent.py)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import guardrails  # noqa: E402
from agent.agent import MessagingAgent, classify_reply  # noqa: E402
from agent.policy import decide  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
SAMPLE = [json.loads(l) for l in (DATA / "sample.jsonl").read_text().splitlines() if l.strip()]


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
    """Returns a non-compliant draft first, compliant second: exercises enforce + repair."""
    def __init__(self):
        self.calls = 0

    def compose(self, record, d, shots, feedback=None):
        self.calls += 1
        body = ("Great for families! See you soon." if self.calls == 1
                else "Hi Taylor—welcome to Oak Ridge! Book a tour Thu or Fri? Reply 1 for Thu, 2 for Fri. Reply STOP to opt out.")
        return {"next_message": {"channel": "email", "send_at": "garbage", "subject": "hi", "body": body,
                                 "cta": {"type": "x", "options": ["Thu", "Fri"]}},
                "next_action": {"type": "start_cadence", "name": "prospect_welcome_short_horizon"},
                "rationale": "test"}


def test_llm_output_is_enforced_and_repaired():
    agent = MessagingAgent(examples=SAMPLE)
    agent.llm = FakeLLM()
    out = agent.run(SAMPLE[0])
    m = out["next_message"]
    assert agent.llm.calls == 2
    assert m["channel"] == "sms" and m["subject"] is None          # sms first consented preference
    assert m["send_at"] == "2025-12-09T09:00:00-06:00"              # policy time wins
    assert m["cta"]["type"] == "schedule_tour"
    assert out["violations"] == []


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
