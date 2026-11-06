"""Message composition: a deterministic template composer (works offline, fast,
always compliant) and an LLM composer that learns tone/structure/decisions from
few-shot examples drawn from the training JSONL.
"""
from __future__ import annotations

import json
import os
from datetime import date

from .policy import OPT_OUT_TEXT, Decision, short_property_name


def OPT_OUT_TEXT_FOR(lang: str, channel: str) -> str:
    return OPT_OUT_TEXT.get(lang, OPT_OUT_TEXT["en"])[channel]

MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
             "septiembre", "octubre", "noviembre", "diciembre"]
AMENITY_PHRASE = {"pool": "pool", "fitness": "24/7 fitness center", "gym": "24/7 fitness center",
                  "dog_park": "dog park", "pet": "pet-friendly amenities", "parking": "covered parking",
                  "coworking": "coworking lounge", "rooftop": "rooftop deck", "ev_charging": "EV charging"}
AMENITY_SHORT = {"fitness": "fitness rooms", "gym": "fitness rooms"}


def _join(items: list[str], conj: str = "and") -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {conj} " + items[-1]


MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
             "septiembre", "octubre", "noviembre", "diciembre"]


def _move_phrase(move, lang: str = "en") -> str | None:
    try:
        d = date.fromisoformat(str(move)[:10])
    except (TypeError, ValueError):
        return None
    third = 0 if d.day <= 10 else 1 if d.day <= 20 else 2
    if lang == "es":
        return f"{['principios', 'mediados', 'finales'][third]} de {MONTHS_ES[d.month - 1]}"
    return f"{['early', 'mid', 'late'][third]}\u2011{d.strftime('%B')}"  # non-breaking hyphen, matches sample style


# Generic phrasing for CTAs other than tours (not in the samples; kept plain and neutral).
CTA_PHRASE = {
    "apply_now": ("Ready to apply? You can start your application online.", "Apply now"),
    "renew_lease": ("Your lease renewal options are ready to review.", "Review renewal"),
    "pay_rent": ("A quick reminder that rent is due soon.", "Pay now"),
    "submit_maintenance": ("Need something fixed? Submit a maintenance request anytime.", "Submit request"),
}

CTA_PHRASE_ES = {
    "apply_now": ("¿Listo para aplicar? Puedes iniciar tu solicitud en línea.", "Aplicar ahora"),
    "renew_lease": ("Tus opciones de renovación de contrato ya están listas.", "Ver renovación"),
    "pay_rent": ("Un recordatorio rápido: la renta vence pronto.", "Pagar ahora"),
    "submit_maintenance": ("¿Necesitas una reparación? Envía una solicitud de mantenimiento cuando quieras.", "Enviar solicitud"),
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
            if es:
                line, _ = CTA_PHRASE_ES.get(d.cta_type, ("Responde con cualquier pregunta—con gusto te ayudamos.", ""))
                body = f"{hola}—un mensaje rápido de {prop}. {line} {d.opt_out_text}"
            else:
                line, _ = CTA_PHRASE.get(d.cta_type, ("Reply with any questions—we’re happy to help.", ""))
                body = f"{hi}—a quick note from {prop}. {line} {d.opt_out_text}"
    elif d.channel == "email":
        am_long = [AMENITY_PHRASE.get(a, a.replace("_", " ")) for a in amenities]
        am_short = [AMENITY_SHORT.get(a, AMENITY_PHRASE.get(a, a.replace("_", " "))) for a in amenities]
        move = _move_phrase(inp.get("move_date_target"), d.language) if d.horizon != "unknown" else None
        link_line = lambda label: [f"{label} → {d.cta_link}"] if d.cta_link else []
        if not tour:
            phrases = CTA_PHRASE_ES if es else CTA_PHRASE
            fallback = (f"Una actualización rápida de {prop}.", "Más información") if es else (f"Here’s a quick update from {prop}.", "Learn more")
            line, label = phrases.get(d.cta_type, fallback)
            subject = f"{prop}—{label}" if d.cta_type in phrases else (f"Novedades de {prop}" if es else f"An update from {prop}")
            lines = [f"{hola if es else hi},", line] + link_line(label) + [d.opt_out_text]
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
        if es:
            ask = "llámanos para reservar una visita" if tour else "llámanos cuando quieras si tienes preguntas"
            body = f"{hola}, te habla el equipo de {prop}—{ask}. {d.opt_out_text}"
        else:
            ask = "call us back to book a tour" if tour else "call us back anytime with questions"
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

You make the workflow decisions too: whether to send, which channel, when (send_at with UTC offset),
and the next_action. Base them on the patterns in the examples. The input also gives you:
- hard_limits: rules you must never break (consented channels, quiet hours, opt-out text per channel).
  Anything that breaks them is rejected and replaced.
- defaults: what the agent would do without you. They are assumptions, not learned rules. Use them only
  when the examples don't show a clearer pattern for this situation.
- cta_by_channel: the reply options or link to use for whichever channel you choose.
Never:
- use a channel without consent or send outside quiet hours
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
    from .policy import ASSUMPTIONS, QUIET_END, QUIET_START, with_channel_and_time
    inp = record.get("input") or {}
    hard_limits = {
        "eligible_channels": d.eligible_channels,
        "quiet_hours_local": f"{QUIET_START:02d}:00-{QUIET_END:02d}:00 {getattr(d.tz, 'key', d.tz)}",
        "send_after": d.last_local.isoformat() if d.last_local else None,
        "opt_out_text_by_channel": {c: OPT_OUT_TEXT_FOR(d.language, c) for c in d.eligible_channels},
        "language": d.language,
    }
    defaults = {"channel": d.channel, "send_at": d.send_at, "next_action": d.next_action, "horizon": d.horizon,
                "basis": {k: v["basis"] for k, v in ASSUMPTIONS.items()}}
    cta_by_channel = {}
    if d.send_at:
        from datetime import datetime as _dt
        for c in d.eligible_channels:
            dc = with_channel_and_time(d, record, c, _dt.fromisoformat(d.send_at))
            cta_by_channel[c] = {"type": dc.cta_type, "options": dc.cta_options or None,
                                 "option_labels": dc.cta_option_labels or None, "link": dc.cta_link,
                                 "tour_days_are_this_week": dc.cta_same_week}
    policy = {
        "hard_limits": hard_limits, "defaults": defaults, "cta_by_channel": cta_by_channel,
        "move_timing": _move_phrase(inp.get("move_date_target"), d.language),
        "first_name": (inp.get("profile") or {}).get("first_name"),
    }
    user = (f"<examples>\n{shots}\n</examples>\n\n"
            f"<decision_context>{json.dumps(policy, ensure_ascii=False)}</decision_context>\n\n"
            f"<record>{json.dumps(_without_expected(record), ensure_ascii=False)}</record>\n\n"
            f"Write the message in language '{d.language}'. Personalize with the first name and any stated "
            f"interests in the record (not the examples' names, days or interests). If first_name is null, "
            f"greet with 'Hi there'. Describe the move date using move_timing exactly. If the channel's "
            f"tour_days_are_this_week is false, the tour days are NEXT week: never say 'this week' about them. "
            f"Output JSON only.")
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
        return _strip_nulls(self.json_call(SYSTEM_PROMPT, build_user_prompt(record, d, examples, feedback), OUTPUT_SCHEMA))

    def json_call(self, system: str, user: str, schema: dict) -> dict:
        import urllib.request

        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "num_ctx": 8192},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return json.loads(data["message"]["content"])


class OpenAIComposer:
    """OpenAI API via the official SDK. Needs OPENAI_API_KEY (an API key from platform.openai.com;
    a ChatGPT subscription does not include API access). Uses strict JSON-schema structured outputs."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI  # imported lazily so other modes don't need the SDK

        self.client = OpenAI()  # reads OPENAI_API_KEY (and OPENAI_BASE_URL if set)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5-mini")

    def compose(self, record: dict, d: Decision, examples: list[dict], feedback: list[str] | None = None) -> dict:
        return _strip_nulls(self.json_call(SYSTEM_PROMPT, build_user_prompt(record, d, examples, feedback), OUTPUT_SCHEMA))

    def json_call(self, system: str, user: str, schema: dict) -> dict:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "result", "strict": True, "schema": schema}},
            max_completion_tokens=4000,
        )
        if self.model.startswith(("gpt-5", "o")):
            kwargs["reasoning_effort"] = "low"  # reasoning models: keep latency down
        else:
            kwargs["temperature"] = 0
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise RuntimeError(f"model refused: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise RuntimeError("model output truncated (length)")
        return json.loads(choice.message.content)


class ClaudeComposer:
    def __init__(self, model: str | None = None, effort: str = "low"):
        import anthropic  # imported lazily so offline mode needs no SDK

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("AGENT_MODEL", "claude-opus-5-5")
        self.effort = effort

    def compose(self, record: dict, d: Decision, examples: list[dict], feedback: list[str] | None = None) -> dict:
        return _strip_nulls(self.json_call(SYSTEM_PROMPT, build_user_prompt(record, d, examples, feedback), OUTPUT_SCHEMA))

    def json_call(self, system: str, user: str, schema: dict) -> dict:
        kwargs = dict(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
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
        return json.loads(text)
