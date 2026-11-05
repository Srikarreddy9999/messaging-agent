"""Deterministic decision layer.

Everything here is a hard rule that must hold regardless of what an LLM says:
consent, channel eligibility, quiet hours, opt-out wording, send-time math.
These are the parts of the problem where "semantically close" is not good
enough (a message on a non-consented channel is a compliance violation).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

CHANNEL_CONSENT_KEY = {"sms": "sms_opt_in", "email": "email_opt_in", "voice": "voice_opt_in"}

# Local send hour per channel, learned from the sample data (sms 09:00, email 10:00).
# Voice is not in the samples; 11:00 keeps it inside business hours.
SEND_HOUR = {"sms": 9, "email": 10, "voice": 11}
QUIET_START, QUIET_END = 8, 21  # TCPA-style window, local time

OPT_OUT_TEXT = {
    "en": {
        "sms": "Reply STOP to opt out.",
        "email": "To opt out of emails, click here or reply STOP.",
        "voice": "To stop these calls, say 'stop' or press 9.",
    },
    "es": {
        "sms": "Responde STOP para cancelar.",
        "email": "Para dejar de recibir correos, haz clic aquí o responde STOP.",
        "voice": "Para no recibir más llamadas, diga 'stop' o presione 9.",
    },
}

# primary_cta in the assertions -> cta.type in the expected output
CTA_TYPE = {"book_tour": "schedule_tour", "schedule_tour": "schedule_tour"}

SHORT_HORIZON_DAYS = 45  # move date within ~6 weeks => short horizon cadence

STOP_WORDS = {"stop", "unsubscribe", "cancel", "end", "quit", "stopall", "opt out", "optout", "remove me"}


@dataclass
class Decision:
    should_send: bool
    channel: str | None
    eligible_channels: list[str]
    send_at: str | None
    language: str
    opt_out_text: str | None
    cta_type: str | None
    cta_options: list[str] = field(default_factory=list)
    cta_option_labels: list[str] = field(default_factory=list)
    cta_link: str | None = None
    cta_same_week: bool = True
    horizon: str = "unknown"  # short | long | unknown
    next_action: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    suppress_reason: str | None = None


def _get(d: dict, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def short_property_name(name: str | None) -> str:
    if not name:
        return "our community"
    return re.sub(r"\s+(Apartments|Apartment Homes|Apts\.?|Residences|Homes|Lofts)$", "", name.strip(), flags=re.I)


def property_link(name: str | None, path: str = "tour") -> str:
    slug = re.sub(r"[^a-z0-9]", "", short_property_name(name).lower()) or "property"
    return f"https://{slug}.example/{path}"


def day_offset_from_task(task_id: str) -> int | None:
    m = re.search(r"day[_-]?(\d+)", task_id or "", re.I)
    return int(m.group(1)) if m else None


def inbound_text(record: dict) -> str | None:
    """Find an inbound reply anywhere obvious in the input (hold-out may include one)."""
    inp = record.get("input", {})
    for key in ("inbound_message", "last_inbound", "reply", "last_reply", "message_text", "inbound"):
        v = inp.get(key) if isinstance(inp, dict) else None
        if isinstance(v, dict):
            v = v.get("body") or v.get("text")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def is_opted_out(record: dict) -> bool:
    consent = record.get("consent", {}) or {}
    inp = record.get("input", {}) or {}
    if consent.get("do_not_contact") or inp.get("do_not_contact") or consent.get("global_opt_out"):
        return True
    text = inbound_text(record)
    return bool(text and text.strip().lower().strip(".! ") in STOP_WORDS)


def decide(record: dict, now: datetime | None = None) -> Decision:
    consent = record.get("consent", {}) or {}
    inp = record.get("input", {}) or {}
    prefs = [c.lower() for c in (record.get("channel_preferences") or [])]
    # Any consented channel not listed in preferences is still eligible, but ranked last.
    for c in CHANNEL_CONSENT_KEY:
        if c not in prefs:
            prefs.append(c)
    language = (inp.get("language") or "en").split("-")[0].lower()
    lang_key = language if language in OPT_OUT_TEXT else "en"
    tz = ZoneInfo(inp.get("timezone") or "America/Chicago")
    reasons: list[str] = []

    eligible = [c for c in prefs if consent.get(CHANNEL_CONSENT_KEY[c]) is True]
    reasons.append(f"consented channels in preference order: {eligible or 'none'}")

    primary_cta = _get(record, "assertions", "constraints", "primary_cta")
    cta_type = CTA_TYPE.get(primary_cta, primary_cta)

    # --- hard stops -------------------------------------------------------
    suppress = None
    if is_opted_out(record):
        suppress = "opted_out"
    elif not eligible:
        suppress = "no_consented_channel"
    elif (record.get("lifecycle_stage") or "").lower() in {"lost", "closed", "do_not_contact", "unsubscribed"}:
        suppress = f"lifecycle_stage={record.get('lifecycle_stage')}"
    if suppress:
        reasons.append(f"suppressed: {suppress}")
        return Decision(
            should_send=False, channel=None, eligible_channels=eligible, send_at=None,
            language=lang_key, opt_out_text=None, cta_type=None, reasons=reasons,
            suppress_reason=suppress,
            next_action={"type": "suppress", "reason": suppress},
        )

    channel = eligible[0]
    reasons.append(f"channel={channel} (first consented preference)")

    # --- timing -----------------------------------------------------------
    last = _parse_ts(inp.get("last_interaction")) or (now or datetime.now(timezone.utc))
    last_local = last.astimezone(tz)
    offset = day_offset_from_task(record.get("task_id", ""))
    offset = max(1, offset) if offset is not None else 1
    send_day = last_local.date() + timedelta(days=offset)
    hour = min(max(SEND_HOUR.get(channel, 10), QUIET_START), QUIET_END - 1)
    send_dt = datetime(send_day.year, send_day.month, send_day.day, hour, 0, tzinfo=tz)
    if now and send_dt <= now:
        send_dt = send_dt + timedelta(days=(now.astimezone(tz).date() - send_day).days + 1)
    send_at = send_dt.isoformat()
    reasons.append(f"send_at={send_at} (last_interaction + {offset}d at {hour}:00 local, inside quiet-hours window)")

    # --- horizon & cadence -------------------------------------------------
    move = _parse_date(inp.get("move_date_target"))
    tid = (record.get("task_id") or "").lower()
    if "short" in tid:
        horizon = "short"
    elif "long" in tid:
        horizon = "long"
    elif move:
        horizon = "short" if (move - last_local.date()).days <= SHORT_HORIZON_DAYS else "long"
    else:
        horizon = "unknown"
    persona = (record.get("persona") or "contact").lower()
    stage = (record.get("lifecycle_stage") or "").lower()
    if stage == "new":
        next_action = {"type": "start_cadence", "name": f"{persona}_welcome_{horizon if horizon != 'unknown' else 'short'}_horizon"}
    else:
        next_action = {"type": "follow_up_in_days", "value": 3}
    reasons.append(f"horizon={horizon} (move date {move}), stage={stage} -> next_action={next_action}")

    # --- CTA ----------------------------------------------------------------
    options, labels, link, same_week = [], [], None, True
    if cta_type == "schedule_tour":
        if channel == "sms":
            # two weekdays at least 2 days after the send day -> numbered reply options
            d = send_day + timedelta(days=2)
            days_es = ["lunes", "martes", "miércoles", "jueves", "viernes"]
            while len(options) < 2:
                if d.weekday() < 5:
                    options.append(d.strftime("%a"))
                    labels.append(days_es[d.weekday()] if lang_key == "es" else d.strftime("%A"))
                    same_week = same_week and d.isocalendar()[1] == send_day.isocalendar()[1]
                d += timedelta(days=1)
        else:
            link = property_link(inp.get("property_name"), "tour")

    return Decision(
        should_send=True, channel=channel, eligible_channels=eligible, send_at=send_at,
        language=lang_key, opt_out_text=OPT_OUT_TEXT[lang_key][channel], cta_type=cta_type,
        cta_options=options, cta_option_labels=labels, cta_link=link, cta_same_week=same_week, horizon=horizon,
        next_action=next_action, reasons=reasons,
    )
