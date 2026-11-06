"""Orchestrator: policy -> compose (LLM or template) -> guardrails -> repair -> output."""
from __future__ import annotations

import dataclasses
import json
import re
import time
from datetime import datetime

from . import guardrails
from .composer import ClaudeComposer, OllamaComposer, OpenAIComposer, compose_template
from .policy import normalize_action, check_next_action, check_send_at, decide, inbound_text, is_stop_request, with_channel_and_time

# --------------------------------------------------------------------------
# Reply classification (the thresholds mention reply_classification_f1)
# --------------------------------------------------------------------------
REPLY_RULES = [
    ("opt_out", r"^\s*(cancel|end|quit)\W*$"),
    ("select_option", r"^\s*[1-9]\s*[.!]?\s*$"),
    # reschedule before schedule_tour: "reschedule my tour" mentions a tour too
    ("reschedule", r"\b(reschedule|another time|different (day|time)|can'?t make it|move it)\b"),
    ("schedule_tour", r"\b(tour|visit|see the (place|unit)|come by|book)\b"),
    ("not_interested", r"\b(not interested|no thanks|found (a|another) place|signed elsewhere)\b"),
    ("pricing_question", r"\b(price|rent|cost|how much|fees?|deposit|specials?)\b"),
    ("availability_question", r"\b(available|availability|units?|floor ?plans?|bedrooms?|\dbr|\d bed)\b"),
    ("question", r"\?\s*$"),
]


def classify_reply(text: str) -> str:
    low = text.lower().strip()
    if is_stop_request(low):
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
    f |= {f"tid={t}" for t in re.split(r"[_\-]", str(r.get("task_id") or "")) if t}
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
                 model: str | None = None, max_repairs: int = 1, now: datetime | None = None,
                 decisions: str = "llm"):
        """llm: None (offline templates) | "ollama" (free, local) | "openai" | "claude"."""
        self.examples = examples or []
        self.llm = {"ollama": OllamaComposer, "openai": OpenAIComposer, "claude": ClaudeComposer}[llm](model=model) if llm else None
        self.max_repairs = max_repairs
        self.now = now
        # "llm": the LLM's channel / send time / next action are used when they pass the hard limits.
        # "rules": the LLM only writes copy; workflow decisions come from the labelled defaults.
        self.decisions = decisions
        self.example_action_types = {((e.get("expected") or {}).get("next_action") or {}).get("type")
                                     for e in self.examples} - {None}

    def run(self, record: dict) -> dict:
        t0 = time.perf_counter()
        d = decide(record, now=self.now)
        trace = list(d.reasons)
        reply = inbound_text(record)
        reply_label = classify_reply(reply) if reply else None
        if reply_label:
            trace.append(f"inbound reply classified as {reply_label!r}")

        message, next_action, rationale, source = None, d.next_action, None, "policy"

        if d.should_send and reply_label and reply_label != "opt_out":
            # The contact just wrote back: answer that, don't send the next cadence step.
            message, next_action, rationale = self._reply_response(record, d, reply_label, reply)
            source = "reply_handler"
            # tour-invite rules (numbered options, tour wording) don't apply to an acknowledgement
            ack = dataclasses.replace(d, cta_type=None, cta_options=[], cta_option_labels=[], cta_link=None)
            states, violations = guardrails.check(record, ack, message)
            return self._result(record, message, next_action, rationale, states, violations,
                                reply_label, source, t0, trace)

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
            d_eff = d
            decided_by = {"channel": "default", "send_at": "default", "next_action": "default"}
            if self.llm:
                shots = select_examples(record, self.examples)
                feedback = None
                for attempt in range(self.max_repairs + 1):
                    try:
                        out = self.llm.compose(record, d, shots, feedback)
                    except Exception as e:  # network/refusal/parse -> fall back to template
                        trace.append(f"llm error: {e!r}; falling back to template")
                        break
                    source, rationale = f"llm(attempt {attempt + 1})", out.get("rationale")
                    message, next_action, d_eff, decided_by = self._accept(out, d, record, trace)
                    if message is None:  # the LLM chose not to send; that is always the safe direction
                        trace.append(f"llm chose not to send; next_action={next_action}")
                        return self._result(record, None, next_action, rationale, [], [], reply_label, source,
                                            t0, trace, decided_by)
                    if guardrails.fix_week_wording(message, d_eff):
                        trace.append(f"tour options {d_eff.cta_options} fall next week: changed 'this week' to 'next week'")
                    _, violations = guardrails.check(record, d_eff, message)
                    if not violations:
                        break
                    trace.append(f"guardrail violations: {violations}")
                    feedback, message, d_eff = violations, None, d
            if message is None:
                message, source = compose_template(record, d), "template"
                next_action, d_eff = d.next_action, d
                decided_by = {"channel": "default", "send_at": "default", "next_action": "default"}
            message["body"] = guardrails.apply_brand_style(message["body"])
            if message.get("subject"):
                message["subject"] = guardrails.apply_brand_style(message["subject"])
            states, violations = guardrails.check(record, d_eff, message)
            if violations and source != "template":
                trace.append(f"final draft failed: {violations}; trying template")
                message, source, next_action, d_eff = compose_template(record, d), "template(repair)", d.next_action, d
                decided_by = {"channel": "default", "send_at": "default", "next_action": "default"}
                message["body"] = guardrails.apply_brand_style(message["body"])
                if message.get("subject"):
                    message["subject"] = guardrails.apply_brand_style(message["subject"])
                states, violations = guardrails.check(record, d, message)
            if violations:
                # Fail safe: nothing that failed validation is ever sent.
                trace.append(f"blocked: every draft failed validation ({violations}); routed to manual review")
                return self._result(record, None, {"type": "manual_review", "reason": "failed_validation"},
                                    "Every draft failed a compliance check, so nothing is sent and a person reviews it.",
                                    [], [], reply_label, "blocked", t0, trace, decided_by, blocked=violations)
            rationale = rationale or self._rationale(record, d)
            return self._result(record, message, next_action, rationale, states, violations, reply_label,
                                source, t0, trace, decided_by)

        states, violations = guardrails.check(record, d, None)
        rationale = rationale or self._rationale(record, d)
        return self._result(record, message, next_action, rationale, states, violations,
                            reply_label, source, t0, trace)

    @staticmethod
    def _result(record, message, next_action, rationale, states, violations, reply_label, source, t0, trace,
                decided_by=None, blocked=None):
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "task_id": record.get("task_id"),
            "next_message": message,
            "next_action": next_action,
            "rationale": rationale,
            "states": states,
            "violations": violations,
            "reply_classification": reply_label,
            "meta": {"source": source, "decided_by": decided_by or {}, "latency_ms": latency_ms,
                     **({"blocked_violations": blocked} if blocked else {}), "trace": trace},
        }

    @staticmethod
    def _reply_response(record: dict, d, label: str, reply: str):
        prof = (record.get("input") or {}).get("profile") or {}
        first = prof.get("first_name")
        es = d.language == "es"
        if label == "not_interested":
            return None, {"type": "close_lead", "reason": "not_interested"}, \
                "Contact said they are not interested; stop the cadence and send nothing more."
        if label == "select_option" and d.cta_options:
            idx = int(re.sub(r"\D", "", reply)) - 1
            day = d.cta_option_labels[idx] if 0 <= idx < len(d.cta_option_labels) else None
        else:
            day = None
        hi = (f"Hola {first}" if first else "Hola") if es else (f"Hi {first}" if first else "Hi there")
        if day:
            body = (f"{hi}—¡perfecto! Te confirmaremos la hora de tu recorrido del {day} en breve." if es
                    else f"{hi}—great choice! We’ll confirm your {day} tour time shortly.")
            action = {"type": "confirm_tour", "day": day}
        else:
            body = (f"{hi}—¡gracias por tu mensaje! Un miembro de nuestro equipo te responderá en breve." if es
                    else f"{hi}—thanks for your message! A member of our leasing team will follow up shortly.")
            action = {"type": "handoff_to_leasing_agent", "reason": label}
        body = guardrails.apply_brand_style(f"{body} {d.opt_out_text}")
        msg = {"channel": d.channel, "send_at": "immediate", "subject": "Re: your message" if d.channel == "email" else None,
               "body": body, "cta": {"type": action["type"]}}
        return msg, action, f"Inbound reply classified as {label}; acknowledge it and {action['type'].replace('_', ' ')}."

    def _accept(self, out: dict, d, record: dict, trace: list[str]):
        """Take the LLM's workflow decisions where they respect the hard limits; fall back per field otherwise."""
        if self.decisions == "rules":
            out = {**out, "next_action": d.next_action}
            if out.get("next_message"):
                out["next_message"] = {**out["next_message"], "channel": d.channel, "send_at": d.send_at}
        decided_by = {}
        action = out.get("next_action")
        problem = check_next_action(action, self.example_action_types)
        if problem:
            trace.append(f"llm next_action rejected ({problem}); using default {d.next_action}")
            action, decided_by["next_action"] = d.next_action, "default"
        else:
            action, decided_by["next_action"] = normalize_action(action), "llm"
        msg = out.get("next_message")
        if msg is None:
            if decided_by["next_action"] == "default" or action.get("type") == "start_cadence":
                action = {"type": "manual_review", "reason": "llm_declined_to_send"}
            return None, action, d, decided_by
        msg = dict(msg)

        channel = msg.get("channel")
        if channel in d.eligible_channels:
            decided_by["channel"] = "llm"
        else:
            trace.append(f"llm channel {channel!r} lacks consent; using default {d.channel}")
            channel, decided_by["channel"] = d.channel, "default"
        when, problem = check_send_at(msg.get("send_at"), d)
        if problem:
            trace.append(f"llm send_at rejected ({problem}); using default {d.send_at}")
            when, decided_by["send_at"] = datetime.fromisoformat(d.send_at), "default"
        else:
            decided_by["send_at"] = "llm"
        d_eff = with_channel_and_time(d, record, channel, when)
        if decided_by["channel"] == "llm" and channel != d.channel:
            trace.append(f"llm chose channel {channel} over default {d.channel}")

        msg["channel"], msg["send_at"] = channel, d_eff.send_at
        if channel in ("sms", "voice"):
            msg["subject"] = None
        body = msg.get("body") or ""
        if d_eff.opt_out_text and not guardrails.OPT_OUT_RE.search(body):
            body = body.rstrip() + ("\n" if channel == "email" else " ") + d_eff.opt_out_text
        if channel in ("sms", "voice"):
            body = " ".join(body.split())  # single line
        msg["body"] = guardrails.apply_brand_style(body)
        if msg.get("subject"):
            msg["subject"] = guardrails.apply_brand_style(msg["subject"])
        cta = {k: v for k, v in (msg.get("cta") or {}).items() if k not in ("options", "link")}
        if d_eff.cta_type:
            cta["type"] = d_eff.cta_type
        if d_eff.cta_options:
            cta["options"] = list(d_eff.cta_options)
        if d_eff.cta_link:
            cta["link"] = d_eff.cta_link
        msg["cta"] = cta
        return msg, action, d_eff, decided_by

    @staticmethod
    def _rationale(record: dict, d) -> str:
        if not d.should_send:
            return f"Suppressed: {d.suppress_reason}."
        prof = (record.get("input") or {}).get("profile") or {}
        bits = [f"{d.channel.upper()} is the first channel in the contact's preferences with explicit consent",
                f"sent {d.send_at} local business hours"]
        if prof.get("amenity_interest"):
            bits.append(f"personalized to stated interests ({', '.join(map(str, prof['amenity_interest']))})")
        bits.append(f"primary CTA {d.cta_type}; {d.horizon}-horizon move -> {json.dumps(d.next_action)}")
        return "; ".join(bits) + "."
