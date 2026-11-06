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



def test_week_wording_and_spanish_move():
    from agent.composer import _move_phrase
    from agent.guardrails import check
    from agent.policy import decide
    assert _move_phrase("2026-01-20", "es") == "mediados de enero"
    assert _move_phrase("2026-01-25") == "late\u2011January"
    # Thursday send -> Mon/Tue options are next week
    rec = {"task_id": "t_day1", "persona": "prospect", "lifecycle_stage": "new",
           "consent": {"sms_opt_in": True}, "channel_preferences": ["sms"],
           "input": {"property_name": "Oak Ridge Apartments", "move_date_target": "2026-01-25",
                     "last_interaction": "2025-12-10T18:00:00Z", "timezone": "America/Chicago",
                     "language": "en", "profile": {"first_name": "Morgan"}},
           "assertions": {"constraints": {"primary_cta": "book_tour", "include_opt_out_instructions": True}}}
    d = decide(rec)
    assert d.cta_options == ["Mon", "Tue"] and not d.cta_same_week
    bad = {"channel": "sms", "subject": None, "cta": {"type": "schedule_tour"},
           "body": "Hi Morgan—tours are available this week. Reply 1 for Mon, 2 for Tue. Reply STOP to opt out."}
    assert any("next week" in v for v in check(rec, d, bad)[1])
    from agent.guardrails import fix_week_wording
    fixed = dict(bad)
    assert fix_week_wording(fixed, d) and "next week" in fixed["body"] and "this week" not in fixed["body"]
    # reverse direction: Tuesday send -> Thu/Fri are this week
    rec_tue = json.loads(json.dumps(rec)); rec_tue["input"]["last_interaction"] = "2025-12-08T16:00:00Z"
    d_tue = decide(rec_tue)
    assert d_tue.cta_options == ["Thu", "Fri"] and d_tue.cta_same_week
    wrong = dict(bad, body="Hi Morgan—tours open la próxima semana and next week. Reply 1 for Thu, 2 for Fri. Reply STOP to opt out.")
    assert fix_week_wording(wrong, d_tue) and "next week" not in wrong["body"] and "esta semana" in wrong["body"]
    good = dict(bad, body=bad["body"].replace("this week", "next week"))
    assert not any("week" in v for v in check(rec, d, good)[1])
    # offline template already gets it right, in both languages
    out = MessagingAgent(examples=SAMPLE).run(rec)
    assert "next week" in out["next_message"]["body"] and out["violations"] == []
    rec_es = json.loads(json.dumps(rec)); rec_es["input"]["language"] = "es"
    body_es = MessagingAgent(examples=SAMPLE).run(rec_es)["next_message"]["body"]
    assert "la próxima semana" in body_es and "January" not in body_es
    d_es = decide(rec_es)
    hi_es = {"channel": "sms", "subject": None, "cta": {"type": "schedule_tour"},
             "body": "Hi Morgan—visita la próxima semana. Responde 1 para lunes, 2 para martes. Responde STOP para cancelar."}
    assert any("Spanish greeting" in v for v in check(rec_es, d_es, hi_es)[1])


def test_edge_cases_file():
    """Every record in data/edge_cases.jsonl gets a valid answer with the expected behaviour (offline)."""
    import io, contextlib
    from run import load_jsonl
    with contextlib.redirect_stderr(io.StringIO()):
        records = load_jsonl(str(DATA / "edge_cases.jsonl"))
    agent = MessagingAgent(examples=SAMPLE)
    out = {}
    for r in records:
        try:
            o = agent.run(r)
        except Exception as e:  # run.py turns these into manual_review; here it must not happen except bad JSON
            assert "__invalid_line__" in r, (r.get("task_id"), e)
            continue
        out[o["task_id"]] = o
    suppressed = {"prospect_no_consent": "no_consented_channel", "sparse_record": "no_consented_channel",
                  "do_not_contact_flag": "opted_out", "lifecycle_closed_lost": "lifecycle_stage=closed_lost",
                  "consent_as_strings_is_not_consent": "no_consented_channel"}
    for tid, reason in suppressed.items():
        assert out[tid]["next_message"] is None and out[tid]["next_action"]["reason"] == reason, tid
    for tid in ("prospect_stop_reply", "reply_spanish_opt_out"):
        assert out[tid]["next_action"]["type"] == "opt_out" and "unsubscribed" in out[tid]["next_message"]["body"], tid
    assert out["reply_not_interested"]["next_message"] is None
    assert out["reply_not_interested"]["next_action"]["type"] == "close_lead"
    assert out["reply_select_option_1"]["next_action"]["type"] == "confirm_tour"
    for tid in ("reply_reschedule", "reply_general_question", "reply_stop_by_is_not_opt_out"):
        assert out[tid]["next_action"]["type"] == "handoff_to_leasing_agent", tid
    for tid in ("late_night_last_interaction_day0", "thursday_send_tour_days_next_week_day1", "friday_send_day1"):
        body = out[tid]["next_message"]["body"]
        assert "next week" in body and "this week" not in body and "Reply 1 for Mon, 2 for Tue" in body, tid
    assert out["prospect_sms_pref_but_only_email_consent"]["next_message"]["channel"] == "email"
    assert out["empty_preferences_uses_consented_channel"]["next_message"]["channel"] == "sms"
    assert out["invalid_timezone_day1"]["next_message"]["send_at"].endswith("-06:00")
    assert "Hi Xander" in out["preferred_name_used"]["next_message"]["body"]
    assert "EV charging" in out["unknown_amenity"]["next_message"]["body"]
    leak = out["pii_and_protected_in_profile"]["next_message"]["body"].lower()
    assert not any(w in leak for w in ("pregnant", "wheelchair", "mexico", "555", "123-45", "maria@"))
    assert out["voice_spanish"]["next_message"]["body"].startswith("Hola")
    assert out["unsupported_language_falls_back_to_english"]["next_message"]["body"].startswith("Hi ")
    for tid, o in out.items():
        m = o["next_message"]
        assert o["violations"] == [], (tid, o["violations"])
        if m and m["send_at"] != "immediate":
            hour = int(m["send_at"][11:13])
            assert 8 <= hour < 21, (tid, m["send_at"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
