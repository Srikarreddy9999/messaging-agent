"""Orchestrator: policy -> compose (LLM or template) -> guardrails -> repair -> output."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime

from . import guardrails
from .composer import ClaudeComposer, OllamaComposer, compose_template
from .policy import STOP_WORDS, decide, inbound_text

# --------------------------------------------------------------------------
# Reply classification (the thresholds mention reply_classification_f1)
# --------------------------------------------------------------------------
REPLY_RULES = [
    ("opt_out", r"^\s*(stop|stopall|unsubscribe|cancel|end|quit|opt ?out|remove me)\W*$"),
    ("select_option", r"^\s*[1-9]\s*$"),
    ("schedule_tour", r"\b(tour|visit|see the (place|unit)|come by|book)\b"),
    ("reschedule", r"\b(reschedule|another time|different (day|time)|can'?t make it)\b"),
    ("not_interested", r"\b(not interested|no thanks|found (a|another) place|signed elsewhere)\b"),
    ("pricing_question", r"\b(price|rent|cost|how much|fees?|deposit|specials?)\b"),
    ("availability_question", r"\b(available|availability|units?|floor ?plans?|bedrooms?|\dbr|\d bed)\b"),
    ("question", r"\?\s*$"),
]


def classify_reply(text: str) -> str:
    low = text.lower().strip()
    if low.strip(".! ") in STOP_WORDS:
        return "opt_out"
    for label, pat in REPLY_RULES:
        if re.search(pat, low):
            return label
    return "other"


# --------------------------------------------------------------------------
# Few-shot retrieval: pick the training records most similar to this one
# --------------------------------------------------------------------------
def _features(r: dict) -> set[str]:
    c = r.get("consent") or {}
    inp = r.get("input") or {}
    f = {f"persona={r.get('persona')}", f"stage={r.get('lifecycle_stage')}",
         f"lang={inp.get('language')}", f"pref0={(r.get('channel_preferences') or [''])[0]}"}
    f |= {f"{k}={v}" for k, v in c.items()}
    f |= {f"inkey={k}" for k in inp}
    f |= {f"profkey={k}" for k in (inp.get("profile") or {})}
    f |= {f"tid={t}" for t in re.split(r"[_\-]", r.get("task_id", "")) if t}
    cta = ((r.get("assertions") or {}).get("constraints") or {}).get("primary_cta")
    f.add(f"cta={cta}")
    return f


def select_examples(record: dict, pool: list[dict], k: int = 4) -> list[dict]:
    target = _features(record)
    cands = [e for e in pool if "expected" in e and e.get("task_id") != record.get("task_id")]
    cands.sort(key=lambda e: len(target & _features(e)) / len(target | _features(e)), reverse=True)
    return cands[:k]


# --------------------------------------------------------------------------
class MessagingAgent:
    def __init__(self, examples: list[dict] | None = None, llm: str | None = None,
                 model: str | None = None, max_repairs: int = 1, now: datetime | None = None):
        """llm: None (offline templates) | "ollama" (free, local) | "claude" (Anthropic API)."""
        self.examples = examples or []
        self.llm = {"ollama": OllamaComposer, "claude": ClaudeComposer}[llm](model=model) if llm else None
        self.max_repairs = max_repairs
        self.now = now

    def run(self, record: dict) -> dict:
        t0 = time.perf_counter()
        d = decide(record, now=self.now)
        trace = list(d.reasons)
        reply = inbound_text(record)
        reply_label = classify_reply(reply) if reply else None
        if reply_label:
            trace.append(f"inbound reply classified as {reply_label!r}")

        message, next_action, rationale, source = None, d.next_action, None, "policy"

        if not d.should_send:
            if d.suppress_reason == "opted_out" and reply_label == "opt_out" and record.get("consent", {}).get("sms_opt_in"):
                # one compliant confirmation is allowed after STOP on SMS
                message = {"channel": "sms", "send_at": "immediate", "subject": None,
                           "body": "You’re unsubscribed and won’t receive more messages. Reply START to resubscribe.",
                           "cta": {"type": "none"}}
                next_action = {"type": "opt_out", "channel": "sms"}
            rationale = f"Not sending marketing: {d.suppress_reason}."
        else:
            violations: list[str] = []
            if self.llm:
                shots = select_examples(record, self.examples)
                feedback = None
                for attempt in range(self.max_repairs + 1):
                    try:
                        out = self.llm.compose(record, d, shots, feedback)
                    except Exception as e:  # network/refusal/parse -> fall back to template
                        trace.append(f"llm error: {e!r}; falling back to template")
                        break
                    message = self._enforce(out.get("next_message"), d)
                    # next_action is a workflow decision: policy owns it; the LLM's proposal is only logged.
                    if out.get("next_action") and out["next_action"] != d.next_action:
                        trace.append(f"llm proposed next_action {out['next_action']}; kept policy {d.next_action}")
                    rationale, source = out.get("rationale"), f"llm(attempt {attempt + 1})"
                    _, violations = guardrails.check(record, d, message)
                    if not violations:
                        break
                    trace.append(f"guardrail violations: {violations}")
                    feedback, message = violations, None
            if message is None:
                message, source = compose_template(record, d), "template"
                next_action = d.next_action
            message["body"] = guardrails.apply_brand_style(message["body"])
            if message.get("subject"):
                message["subject"] = guardrails.apply_brand_style(message["subject"])

        states, violations = guardrails.check(record, d, message if d.should_send else None)
        if violations and d.should_send:
            # last resort: deterministic template is always compliant
            message, source, next_action = compose_template(record, d), "template(repair)", d.next_action
            message["body"] = guardrails.apply_brand_style(message["body"])
            states, violations = guardrails.check(record, d, message)

        rationale = rationale or self._rationale(record, d)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "task_id": record.get("task_id"),
            "next_message": message,
            "next_action": next_action,
            "rationale": rationale,
            "states": states,
            "violations": violations,
            "reply_classification": reply_label,
            "meta": {"source": source, "latency_ms": latency_ms, "trace": trace},
        }

    @staticmethod
    def _enforce(msg: dict | None, d) -> dict | None:
        """Hard constraints win over the LLM."""
        if msg is None:
            return None
        msg = dict(msg)
        # Channel is a policy decision (first consented preference), not a copywriting one.
        msg["channel"] = d.channel
        msg["send_at"] = d.send_at
        if msg["channel"] == "sms":
            msg["subject"] = None
        body = msg.get("body") or ""
        if d.opt_out_text and d.opt_out_text.lower() not in body.lower() and "stop" not in body.lower():
            msg["body"] = body.rstrip() + ("\n" if msg["channel"] == "email" else " ") + d.opt_out_text
        if msg["channel"] in ("sms", "voice"):
            msg["body"] = " ".join(msg["body"].split())  # single line
        msg["body"] = guardrails.apply_brand_style(msg["body"])
        if msg.get("subject"):
            msg["subject"] = guardrails.apply_brand_style(msg["subject"])
        cta = msg.get("cta") or {}
        if d.cta_type:
            cta["type"] = d.cta_type
        if d.cta_options:
            cta["options"] = list(d.cta_options)
            cta.pop("link", None)
        if d.cta_link:
            cta["link"] = d.cta_link
            cta.pop("options", None)
        msg["cta"] = cta
        return msg

    @staticmethod
    def _rationale(record: dict, d) -> str:
        if not d.should_send:
            return f"Suppressed: {d.suppress_reason}."
        prof = (record.get("input") or {}).get("profile") or {}
        bits = [f"{d.channel.upper()} is the first channel in the contact's preferences with explicit consent",
                f"sent {d.send_at} local business hours"]
        if prof.get("amenity_interest"):
            bits.append(f"personalized to stated interests ({', '.join(prof['amenity_interest'])})")
        bits.append(f"primary CTA {d.cta_type}; {d.horizon}-horizon move -> {json.dumps(d.next_action)}")
        return "; ".join(bits) + "."
