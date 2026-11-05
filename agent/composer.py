"""Message composition: a deterministic template composer (works offline, fast,
always compliant) and an LLM composer that learns tone/structure/decisions from
few-shot examples drawn from the training JSONL.
"""
from __future__ import annotations

import json
import os
from datetime import date

from .policy import Decision, short_property_name

MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
             "septiembre", "octubre", "noviembre", "diciembre"]
AMENITY_PHRASE = {"pool": "pool", "fitness": "24/7 fitness center", "gym": "24/7 fitness center",
                  "dog_park": "dog park", "pet": "pet-friendly amenities", "parking": "covered parking",
                  "coworking": "coworking lounge", "rooftop": "rooftop deck"}
AMENITY_SHORT = {"fitness": "fitness rooms", "gym": "fitness rooms"}


def _join(items: list[str], conj: str = "and") -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {conj} " + items[-1]


def _move_phrase(move) -> str | None:
    try:
        d = date.fromisoformat(str(move)[:10])
    except (TypeError, ValueError):
        return None
    part = "early" if d.day <= 10 else "mid" if d.day <= 20 else "late"
    return f"{part}\u2011{d.strftime('%B')}"  # non-breaking hyphen, matches sample style


# Generic phrasing for CTAs other than tours (not in the samples; kept plain and neutral).
CTA_PHRASE = {
    "apply_now": ("Ready to apply? You can start your application online.", "Apply now"),
    "renew_lease": ("Your lease renewal options are ready to review.", "Review renewal"),
    "pay_rent": ("A quick reminder that rent is due soon.", "Pay now"),
    "submit_maintenance": ("Need something fixed? Submit a maintenance request anytime.", "Submit request"),
}


def _amenity_key(a: str) -> str:
    a = str(a).lower()
    return "fitness" if a in ("gym", "fitness_center", "fitness") else a


# ---------------------------------------------------------------------------
# Template composer
# ---------------------------------------------------------------------------
def compose_template(record: dict, d: Decision) -> dict:
    inp = record.get("input") or {}
    prof = inp.get("profile") or {}
    first = prof.get("preferred_name") or prof.get("first_name")
    hi = f"Hi {first}" if first else "Hi there"
    hola = hi.replace("Hi there", "Hola").replace("Hi ", "Hola ")
    prop = short_property_name(inp.get("property_name"))
    stage = str(record.get("lifecycle_stage") or "").lower()
    amenities = [_amenity_key(a) for a in (prof.get("amenity_interest") or []) if a]
    es = d.language == "es"
    tour = d.cta_type == "schedule_tour"

    subject = None
    cta: dict = {"type": d.cta_type} if d.cta_type else {}

    if d.channel == "sms":
        if tour and d.cta_options:
            o1, o2 = d.cta_options
            l1, l2 = d.cta_option_labels
            when_en = "this week" if d.cta_same_week else "next week"
            when_es = "esta semana" if d.cta_same_week else "la próxima semana"
            if es:
                opener = f"{hola}—¡te damos la bienvenida a {prop}!" if stage == "new" else f"{hola}—gracias por tu interés en {prop}."
                body = f"{opener} Hay recorridos disponibles {when_es}. ¿Te gustaría reservar el {l1} o el {l2}? Responde 1 para {l1}, 2 para {l2}. {d.opt_out_text}"
            else:
                opener = f"{hi}—welcome to {prop}!" if stage == "new" else f"{hi}—still planning your move? {prop} has tours open."
                body = f"{opener} Tours are available {when_en}. Would you like to book a time on {l1} or {l2}? Reply 1 for {o1}, 2 for {o2}. {d.opt_out_text}"
            cta["options"] = list(d.cta_options)
        else:
            line, _ = CTA_PHRASE.get(d.cta_type, ("Reply with any questions—we’re happy to help.", ""))
            body = f"{hi}—a quick note from {prop}. {line} {d.opt_out_text}"
    elif d.channel == "email":
        am_long = [AMENITY_PHRASE.get(a, a.replace("_", " ")) for a in amenities]
        am_short = [AMENITY_SHORT.get(a, AMENITY_PHRASE.get(a, a.replace("_", " "))) for a in amenities]
        move = _move_phrase(inp.get("move_date_target")) if d.horizon != "unknown" else None
        link_line = lambda label: [f"{label} → {d.cta_link}"] if d.cta_link else []
        if not tour:
            line, label = CTA_PHRASE.get(d.cta_type, (f"Here’s a quick update from {prop}.", "Learn more"))
            subject = f"{prop}—{label}" if d.cta_type in CTA_PHRASE else f"An update from {prop}"
            lines = [f"{hi},", line] + link_line(label) + [d.opt_out_text]
        elif es:
            subject = f"Visita {prop}—conoce lo que te interesa"
            lines = [f"{hola},", f"Reserva una visita esta semana para comparar planos en {prop}."] + link_line("Reservar") + [d.opt_out_text]
        else:
            subject = f"Tour {prop}—See the {' & '.join(am_short)} you asked about" if am_short else f"Tour {prop}—Find your next home"
            article = "an" if move and move[0] in "aeiou" else "a"
            intro = f"Since you’re planning {article} {move} move, " if move else ""
            look = f"here’s a quick look at our {_join(am_long)}." if am_long else f"here’s a quick look at {prop}."
            look = look[0].upper() + look[1:] if not intro else look
            lines = [f"{hi},", f"{intro}{look} Book a visit this week to compare floor plans."] + link_line("Book now") + [d.opt_out_text]
        body = "\n".join(lines)
        if d.cta_link:
            cta["link"] = d.cta_link
    else:  # voice script
        ask = "call us back to book a tour this week" if tour else "call us back anytime with questions"
        body = f"{hi}, this is the team at {prop}—{ask}. {d.opt_out_text}"
        cta = {"type": d.cta_type or "callback"}

    return {"channel": d.channel, "send_at": d.send_at, "subject": subject, "body": body, "cta": cta}


# ---------------------------------------------------------------------------
# LLM composer (Claude)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the messaging brain of a property-management CRM. For each contact record you decide
whether to send a message, on which channel, when, what it says, and what happens next.

You learn the house style and decision patterns ONLY from the worked examples provided (input -> expected).
Mirror their structure closely: greeting with first name, em dashes, short sentences, a single clear CTA,
opt-out line last. SMS has no subject and uses numbered reply options; email has a subject and a link CTA.

A deterministic policy engine has already computed the hard constraints (consented channels, send time,
opt-out wording, CTA options/link). Treat them as authoritative unless an example clearly shows a different
pattern for this exact situation. Never:
- use a channel without consent
- mention or allude to protected characteristics (race, color, religion, sex, national origin,
  familial status/children, disability, age) or describe who a property is "ideal for"
- include personal data beyond the first name and stated interests (no email/phone/address/last name)
- omit the opt-out instruction
If there is no compliant way to send, set next_message to null and choose a suppress-type next_action.
Return only the JSON object described by the schema."""

CTA_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string"},
        "options": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        "link": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["type", "options", "link"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "next_message": {"anyOf": [{
            "type": "object",
            "properties": {
                "channel": {"type": "string", "enum": ["sms", "email", "voice"]},
                "send_at": {"type": "string"},
                "subject": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "body": {"type": "string"},
                "cta": CTA_SCHEMA,
            },
            "required": ["channel", "send_at", "subject", "body", "cta"],
            "additionalProperties": False,
        }, {"type": "null"}]},
        "next_action": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "value": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["type", "name", "value", "reason"],
            "additionalProperties": False,
        },
        "rationale": {"type": "string"},
    },
    "required": ["next_message", "next_action", "rationale"],
    "additionalProperties": False,
}


def _strip_nulls(obj):
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None or k in ("subject", "next_message")}
    if isinstance(obj, list):
        return [_strip_nulls(v) for v in obj]
    return obj


def _without_expected(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "expected"}


def build_user_prompt(record: dict, d: Decision, examples: list[dict], feedback: list[str] | None = None) -> str:
    """Same prompt for every backend: few-shot examples + policy constraints + the new record."""
    shots = "\n\n".join(
        f"<example>\n<input>{json.dumps(_without_expected(e), ensure_ascii=False)}</input>\n"
        f"<expected>{json.dumps(e['expected'], ensure_ascii=False)}</expected>\n</example>"
        for e in examples
    )
    policy = {
        "should_send": d.should_send, "channel": d.channel, "eligible_channels": d.eligible_channels,
        "send_at": d.send_at, "language": d.language, "opt_out_text": d.opt_out_text,
        "cta": {"type": d.cta_type, "options": d.cta_options or None, "option_labels": d.cta_option_labels or None,
                "link": d.cta_link, "tour_days_are_this_week": d.cta_same_week},
        "horizon": d.horizon, "suggested_next_action": d.next_action, "suppress_reason": d.suppress_reason,
        "move_timing": _move_phrase((record.get("input") or {}).get("move_date_target")),
        "first_name": ((record.get("input") or {}).get("profile") or {}).get("first_name"),
    }
    user = (f"<examples>\n{shots}\n</examples>\n\n"
            f"<policy_engine_output>{json.dumps(policy, ensure_ascii=False)}</policy_engine_output>\n\n"
            f"<record>{json.dumps(_without_expected(record), ensure_ascii=False)}</record>\n\n"
            f"Write the message in language '{d.language}'. Personalize with the first name and any stated "
            f"interests in the record (not the examples' names, days or interests). If first_name is null, "
            f"greet with 'Hi there'. Describe the move date using move_timing exactly. Output JSON only.")
    if feedback:
        user += "\n\nYour previous draft failed these compliance checks; fix them:\n- " + "\n- ".join(feedback)
    return user


class OllamaComposer:
    """Free, local LLM via Ollama's HTTP API (stdlib only, no key). Uses Ollama structured outputs
    (`format` = JSON schema) so the response always parses."""

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 120):
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.timeout = timeout

    def compose(self, record: dict, d: Decision, examples: list[dict], feedback: list[str] | None = None) -> dict:
        import urllib.request

        payload = {
            "model": self.model,
            "stream": False,
            "format": OUTPUT_SCHEMA,
            "options": {"temperature": 0, "num_ctx": 8192},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": build_user_prompt(record, d, examples, feedback)}],
        }
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return _strip_nulls(json.loads(data["message"]["content"]))


class ClaudeComposer:
    def __init__(self, model: str | None = None, effort: str = "low"):
        import anthropic  # imported lazily so offline mode needs no SDK

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("AGENT_MODEL", "claude-opus-5-5")
        self.effort = effort

    def compose(self, record: dict, d: Decision, examples: list[dict], feedback: list[str] | None = None) -> dict:
        user = build_user_prompt(record, d, examples, feedback)
        kwargs = dict(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        )
        if self.model in {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}:
            # Server-side refusal fallback: on a safety decline the API retries on a fallback model.
            resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)

        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {getattr(resp, 'stop_details', None)}")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("model output truncated (max_tokens)")
        text = next(b.text for b in resp.content if b.type == "text")
        out = json.loads(text)
        return _strip_nulls(out)
