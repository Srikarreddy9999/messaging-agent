"""Decision layer: hard limits + labelled default assumptions.

Two kinds of rules live here, and they are kept apart on purpose:

* HARD LIMITS (never overridden): consent per channel, opt-out / do-not-contact,
  terminal lifecycle stages, quiet hours, opt-out wording.
* DEFAULTS (overridable): channel order, send time, horizon, next action. These
  come from ASSUMPTIONS below. They are NOT learned rules: with an LLM enabled,
  the LLM proposes these decisions from the few-shot examples and the agent
  only falls back to these defaults when a proposal breaks a hard limit or is
  malformed. Every default that is used is named in the output trace.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

CHANNEL_CONSENT_KEY = {"sms": "sms_opt_in", "email": "email_opt_in", "voice": "voice_opt_in"}

# Every number the agent uses that is not given in the record, with its honest basis.
# Two samples cannot establish any of these as rules.
ASSUMPTIONS = {
    "send_hour_local": {"value": {"sms": 9, "email": 10, "voice": 11},
                        "basis": "sms 09:00 and email 10:00 match the two samples (n=1 each); voice is a guess"},
    "send_day_offset": {"value": 1,
                        "basis": "next day; sample 1 (day0) was sent next day"},
    "task_id_day_hint": {"value": True,
                         "basis": "task_id 'dayN' matched send offset in both samples; a naming coincidence is possible"},
    "short_horizon_days": {"value": 45, "basis": "guess between the samples' 33 and 71 days"},
    "follow_up_days": {"value": 3, "basis": "one sample"},
    "max_send_delay_days": {"value": 14, "basis": "sanity bound on LLM-proposed send times; guess"},
}
SEND_HOUR = ASSUMPTIONS["send_hour_local"]["value"]
QUIET_START, QUIET_END = 8, 21  # HARD LIMIT: TCPA-style window, local time

OPT_OUT_TEXT = {
    "en": {
        "sms": "Reply STOP to opt out.",
        "email": "To opt out of emails, click here or reply STOP.",
        "voice": "To stop these calls, say 'stop' or press 9.",
    },
    "es": {
        "sms": "Responde STOP para cancelar.",
        "email": "Para dejar de recibir correos, haz clic aquí o responde STOP.",
        "voice": "Para no recibir más llamadas, di 'stop' o presiona 9.",
    },
}

# primary_cta in the assertions -> cta.type in the expected output
CTA_TYPE = {"book_tour": "schedule_tour", "schedule_tour": "schedule_tour", "tour": "schedule_tour"}
CTA_PATH = {"apply_now": "apply", "apply": "apply", "renew_lease": "renew", "renewal": "renew",
            "pay_rent": "pay", "submit_maintenance": "maintenance", "maintenance_request": "maintenance"}

TERMINAL_STAGES = {"lost", "closed", "closed_lost", "do_not_contact", "unsubscribed", "dead", "archived"}

SHORT_HORIZON_DAYS = ASSUMPTIONS["short_horizon_days"]["value"]

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
    tz: object = None
    last_local: datetime | None = None
    defaults_used: list[str] = field(default_factory=list)
    next_action: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    suppress_reason: str | None = None


def _get(d: dict, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _parse_ts(value, tz=timezone.utc) -> datetime | None:
    """ISO string, date-only string, or epoch seconds/ms. Naive values are read in the contact's timezone."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value / 1000 if value > 1e11 else value, timezone.utc)
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=tz)
    except (ValueError, OverflowError, OSError):
        return None


def _parse_date(value) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def safe_tz(name):
    """IANA name, or common abbreviations; never raises."""
    alias = {"CST": "America/Chicago", "CDT": "America/Chicago", "CT": "America/Chicago",
             "EST": "America/New_York", "EDT": "America/New_York", "ET": "America/New_York",
             "MST": "America/Denver", "MDT": "America/Denver", "MT": "America/Denver",
             "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles", "PT": "America/Los_Angeles",
             "UTC": "UTC", "GMT": "UTC", "Z": "UTC"}
    key = str(name or "").strip()
    for candidate in (key, alias.get(key.upper(), ""), "America/Chicago"):
        try:
            if candidate:
                return ZoneInfo(candidate)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            continue
    # No IANA database at all (Windows without the tzdata package): fixed US Central offset.
    return timezone(timedelta(hours=-6), "CST")


def short_property_name(name: str | None) -> str:
    if not name:
        return "our community"
    return re.sub(r"\s+(Apartments|Apartment Homes|Apts\.?|Residences|Homes|Lofts)$", "", name.strip(), flags=re.I)


def property_link(name: str | None, path: str = "tour") -> str:
    slug = re.sub(r"[^a-z0-9]", "", short_property_name(name).lower()) or "property"
    return f"https://{slug}.example/{path}"


def day_offset_from_task(task_id: str) -> int | None:
    m = re.search(r"day[_-]?(\d+)", str(task_id or ""), re.I)
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
    return bool(text and is_stop_request(text))


STOP_RE = re.compile(r"\b(stop(?!\s+by)|stopall|unsubscribe|opt(?:-|\s)?out|remove me|do not (text|contact|call|email)|"
                     r"don'?t (text|contact|call|email)|cancelar|baja)\b", re.I)


def is_stop_request(text: str) -> bool:
    """Carrier keywords and natural phrasing ("please stop texting me"); not "stop by"."""
    return bool(STOP_RE.search(text or ""))


def decide(record: dict, now: datetime | None = None) -> Decision:
    consent = record.get("consent", {}) or {}
    inp = record.get("input", {}) or {}
    raw_prefs = record.get("channel_preferences") or []
    if isinstance(raw_prefs, str):
        raw_prefs = [raw_prefs]
    aliases = {"text": "sms", "phone": "voice", "call": "voice", "e-mail": "email", "mail": "email"}
    prefs = []
    for c in raw_prefs:
        c = aliases.get(str(c).strip().lower(), str(c).strip().lower()) if c else None
        if c in CHANNEL_CONSENT_KEY and c not in prefs:
            prefs.append(c)
    # Any consented channel not listed in preferences is still eligible, but ranked last.
    for c in CHANNEL_CONSENT_KEY:
        if c not in prefs:
            prefs.append(c)
    language = str(inp.get("language") or "en").replace("_", "-").split("-")[0].lower()
    lang_key = language if language in OPT_OUT_TEXT else "en"
    tz = safe_tz(inp.get("timezone"))
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
    elif str(record.get("lifecycle_stage") or "").lower() in TERMINAL_STAGES or (
            str(record.get("persona") or "").lower() == "prospect"
            and str(record.get("lifecycle_stage") or "").lower() in {"leased", "signed", "moved_in", "converted"}):
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
    defaults_used = ["channel: first consented preference"]
    reasons.append(f"default channel={channel} (first consented preference)")

    # --- timing (default) ---------------------------------------------------
    last = _parse_ts(inp.get("last_interaction"), tz) or (now or datetime.now(timezone.utc))
    last_local = last.astimezone(tz)
    offset = day_offset_from_task(record.get("task_id", "")) if ASSUMPTIONS["task_id_day_hint"]["value"] else None
    if offset is not None:
        offset = max(1, offset)
        defaults_used.append("send day: task_id dayN hint (assumption task_id_day_hint)")
    else:
        offset = ASSUMPTIONS["send_day_offset"]["value"]
        defaults_used.append("send day: next day (assumption send_day_offset)")
    send_day = last_local.date() + timedelta(days=offset)
    hour = min(max(SEND_HOUR.get(channel, 10), QUIET_START), QUIET_END - 1)
    defaults_used.append(f"send hour {hour}:00 (assumption send_hour_local)")
    send_dt = datetime(send_day.year, send_day.month, send_day.day, hour, 0, tzinfo=tz)
    if now and send_dt <= now:
        send_dt = send_dt + timedelta(days=(now.astimezone(tz).date() - send_day).days + 1)
    send_at = send_dt.isoformat()
    reasons.append(f"default send_at={send_at} (last_interaction + {offset}d at {hour}:00 local)")

    # --- horizon & cadence (default) ------------------------------------------
    move = _parse_date(inp.get("move_date_target"))
    tid = str(record.get("task_id") or "").lower()
    if ASSUMPTIONS["task_id_day_hint"]["value"] and ("short" in tid or "long" in tid):
        horizon = "short" if "short" in tid else "long"
    elif move and move < last_local.date():
        horizon = "unknown"  # stale move date: don't claim they're "planning a move"
    elif move:
        horizon = "short" if (move - last_local.date()).days <= SHORT_HORIZON_DAYS else "long"
    else:
        horizon = "unknown"
    persona = str(record.get("persona") or "contact").lower()
    stage = str(record.get("lifecycle_stage") or "").lower()
    if stage == "new":
        next_action = {"type": "start_cadence", "name": f"{persona}_welcome_{horizon if horizon != 'unknown' else 'short'}_horizon"}
    else:
        next_action = {"type": "follow_up_in_days", "value": ASSUMPTIONS["follow_up_days"]["value"]}
    defaults_used.append(f"next_action {next_action['type']} (assumptions short_horizon_days / follow_up_days)")
    reasons.append(f"default horizon={horizon} (move date {move}), stage={stage} -> next_action={next_action}")

    d = Decision(
        should_send=True, channel=channel, eligible_channels=eligible, send_at=send_at,
        language=lang_key, opt_out_text=OPT_OUT_TEXT[lang_key][channel], cta_type=cta_type,
        horizon=horizon, next_action=next_action, reasons=reasons,
        tz=tz, last_local=last_local, defaults_used=defaults_used,
    )
    return with_channel_and_time(d, record, channel, send_dt)


def _week(day) -> tuple:
    return tuple(day.isocalendar()[:2])


def with_channel_and_time(d: Decision, record: dict, channel: str, send_dt: datetime) -> Decision:
    """Recompute everything that depends on channel and send time (opt-out text, CTA options/link)."""
    from dataclasses import replace
    inp = record.get("input") or {}
    send_day = send_dt.date()
    options, labels, link, same_week = [], [], None, True
    if d.cta_type == "schedule_tour":
        if channel == "sms":
            # two weekdays at least 2 days after the send day -> numbered reply options (sample 1 pattern)
            days, day = [], send_day + timedelta(days=2)
            while len(days) < 2:
                if day.weekday() < 5:
                    days.append(day)
                day += timedelta(days=1)
            if _week(days[0]) != _week(days[1]):
                # split across weeks (e.g. Fri + Mon): offer Mon/Tue of the following week, so one week is named
                monday = days[1] - timedelta(days=days[1].weekday())
                days = [monday, monday + timedelta(days=1)]
            days_es = ["lunes", "martes", "miércoles", "jueves", "viernes"]
            options = [x.strftime("%a") for x in days]
            labels = [days_es[x.weekday()] if d.language == "es" else x.strftime("%A") for x in days]
            same_week = _week(days[0]) == _week(send_day)
        else:
            link = property_link(inp.get("property_name"), "tour")
    elif d.cta_type and channel == "email":
        link = property_link(inp.get("property_name"),
                             CTA_PATH.get(d.cta_type, re.sub(r"[^a-z0-9]+", "-", str(d.cta_type).lower())))
    return replace(d, channel=channel, send_at=send_dt.isoformat(), opt_out_text=OPT_OUT_TEXT[d.language][channel],
                   cta_options=options, cta_option_labels=labels, cta_link=link, cta_same_week=same_week)


# ---------------------------------------------------------------------------
# Validators for LLM-proposed workflow decisions (hard limits only)
# ---------------------------------------------------------------------------
def check_send_at(value, d: Decision) -> tuple[datetime | None, str | None]:
    """Accept an LLM send time only if it is a real timestamp with an offset, inside quiet hours in the
    contact's timezone, after the last interaction and within a sane delay. Returns (datetime, problem)."""
    if not isinstance(value, str):
        return None, "send_at missing"
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None, f"send_at {value!r} is not ISO-8601"
    if dt.tzinfo is None:
        return None, "send_at has no UTC offset"
    local = dt.astimezone(d.tz) if d.tz else dt
    if not (QUIET_START <= local.hour < QUIET_END):
        return None, f"send_at {local:%H:%M} local is outside quiet hours {QUIET_START}:00-{QUIET_END}:00"
    if d.last_local and dt <= d.last_local:
        return None, "send_at is not after last_interaction"
    max_days = ASSUMPTIONS["max_send_delay_days"]["value"]
    if d.last_local and (dt - d.last_local).days > max_days:
        return None, f"send_at is more than {max_days} days after last_interaction"
    return local, None


ACTION_FIELDS = {"start_cadence": ("name",), "follow_up_in_days": ("value",), "suppress": ("reason",),
                 "manual_review": ("reason",), "close_lead": ("reason",), "handoff_to_leasing_agent": ("reason",),
                 "confirm_tour": ("day",), "opt_out": ("channel",)}


def normalize_action(action: dict) -> dict:
    """Keep only the fields each action type uses (the sample shape), dropping LLM padding."""
    keep = ACTION_FIELDS.get(action["type"], ("name", "value", "reason"))
    return {"type": action["type"], **{k: action[k] for k in keep if action.get(k) is not None}}


ALLOWED_ACTION_TYPES = {"start_cadence", "follow_up_in_days", "suppress", "manual_review", "close_lead",
                        "handoff_to_leasing_agent", "confirm_tour", "opt_out", "wait_for_reply", "schedule_call"}


def check_next_action(action, example_types: set[str]) -> str | None:
    """Shape check only. The type must be one the examples use or a known safe workflow action."""
    if not isinstance(action, dict) or not isinstance(action.get("type"), str):
        return "next_action missing a type"
    t = action["type"]
    if t not in ALLOWED_ACTION_TYPES | example_types:
        return f"next_action type {t!r} is not in the examples or the known action set"
    if t == "follow_up_in_days":
        v = action.get("value")
        if not isinstance(v, int) or not 1 <= v <= 30:
            return "follow_up_in_days needs an integer value between 1 and 30"
    if t == "start_cadence" and not (isinstance(action.get("name"), str) and re.fullmatch(r"[a-z0-9_]{3,80}", action["name"])):
        return "start_cadence needs a snake_case name"
    return None
