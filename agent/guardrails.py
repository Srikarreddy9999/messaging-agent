"""Post-generation checks. Every outgoing message passes through here.

Produces the `required_states` the test cases assert on
(consent_verified, fair_housing_check_passed, brand_style_applied) plus a
list of violations. Any violation triggers a repair (LLM retry with feedback,
then deterministic template as a last resort).
"""
from __future__ import annotations

import re

# Fair Housing Act: never steer by race, color, religion, sex, national origin,
# familial status or disability (plus common state-level additions).
# Steering phrases, not bare words: "White Oak Apartments", a contact named Christian, a senior
# leasing consultant or a kids' play area are all fine. Names and the property name are also
# masked before matching (see _mask_names).
_GROUPS = (r"(famil(y|ies)|couples?|singles?|young professionals|students|retirees|seniors|kids|children|"
           r"christians?|catholics?|jews|jewish|muslims?|hindus?|immigrants?|men|women|bachelors?)")
FAIR_HOUSING_PATTERNS = [
    r"\bfamil(y|ies)[- ]friendly\b", rf"\b(perfect|ideal|great|best|suited|suitable|designed) for {_GROUPS}\b",
    r"\bno (kids|children|families|pets and kids)\b", r"\badults?[- ]only\b", r"\bmature (couple|person|individuals|adults?)\b",
    r"\bempty[- ]nesters?\b", r"\b(christian|catholic|jewish|muslim|hindu) (community|neighborhood|area|residents|families)\b",
    r"\b(white|black|hispanic|latino|latina|asian|ethnic|integrated) (neighborhood|community|area|tenants|residents|families)\b",
    r"\b(no|not suitable for|not accessible to) (wheelchairs?|disabled|handicapped)\b", r"\bable[- ]bodied\b",
    r"\bpregnan(t|cy)\b", r"\bno section 8\b", r"\bexclusive (neighborhood|community)\b", r"\bsafe neighborhood\b",
    r"\bnationality\b", r"\benglish[- ]speaking only\b", r"\bbachelor pad\b", r"\bman cave\b",
]


def _mask_names(text: str, record: dict) -> str:
    inp = record.get("input") or {}
    prof = inp.get("profile") or {}
    for v in (prof.get("first_name"), prof.get("preferred_name"), prof.get("last_name"),
              inp.get("property_name"), prof.get("city_interest")):
        if v:
            text = re.sub(re.escape(str(v)), " ", text, flags=re.I)
            short = re.sub(r"\s+(Apartments|Apartment Homes|Apts\.?|Residences|Homes|Lofts)$", "", str(v), flags=re.I)
            text = re.sub(re.escape(short), " ", text, flags=re.I)
    return text


PII_PATTERNS = {
    "email_address": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "phone_number": r"(?<!\d)(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "card_number": r"\b(?:\d[ -]?){13,16}\b",
    "street_address": r"\b\d{2,6}\s+[A-Z][a-z]+\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Ln|Lane|Dr|Drive)\b",
}

# Profile keys that are fine to reference (they are what personalization is for).
ALLOWED_PROFILE_KEYS = {"first_name", "city_interest", "amenity_interest", "unit_interest", "bedrooms",
                        "floor_plan_interest", "preferred_name", "budget_range"}

AMENITY_TERMS = {"pool": r"\b(pool|piscina|alberca)\b", "fitness": r"\b(fitness|gym|gimnasio)\b",
                 "dog": r"\b(dog park|parque para perros)\b", "rooftop": r"\brooftop\b"}

SMS_MAX_CHARS = 320  # two segments
# a bare "stop" is not enough ("stop by for a tour!")
OPT_OUT_RE = re.compile(r"(reply|text|send|responde|say|diga|di)\s+['\"]?stop|\bopt[- ]?out\b|\bunsubscribe\b|"
                        r"\bstop to (opt|unsubscribe|end)|\bpress(ione|iona|ionar)? 9\b|\bcancelar\b", re.I)


def apply_brand_style(text: str) -> str:
    """House style seen in the samples: em dashes, curly apostrophes, single spaces, no shouting."""
    if not text:
        return text
    text = re.sub(r"\s+-\s+|\s*--\s*", "—", text)
    text = re.sub(r"(\w)'(\w)", "\\1’\\2", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"!{2,}", "!", text)
    return text.strip()


def fix_week_wording(message: dict, decision) -> bool:
    """The samples say 'this week'; when the offered tour days fall next week, correct that phrase in place.
    Deterministic, like brand style: a factual fix that doesn't need another LLM round trip."""
    body = message.get("body") or ""
    if message.get("channel") != "sms" or not decision.cta_options or decision.cta_same_week:
        return False
    fixed = re.sub(r"\bthis week\b", "next week", body, flags=re.I)
    fixed = re.sub(r"\besta semana\b", "la próxima semana", fixed, flags=re.I)
    message["body"] = fixed
    return fixed != body


def _strings(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v, prefix)
    elif isinstance(obj, str):
        yield prefix, obj


def check(record: dict, decision, message: dict | None) -> tuple[list[str], list[str]]:
    """Return (states_passed, violations)."""
    states, violations = [], []
    if message is None:
        # Not sending is always compliant; consent was still evaluated.
        return ["consent_verified", "fair_housing_check_passed", "brand_style_applied", "no_pii_leak"], []

    body = message.get("body") or ""
    subject = message.get("subject") or ""
    text = f"{subject}\n{body}"
    low = text.lower()
    constraints = (record.get("assertions") or {}).get("constraints") or {}

    # consent
    if message.get("channel") in decision.eligible_channels:
        states.append("consent_verified")
    else:
        violations.append(f"channel {message.get('channel')} lacks consent")

    # fair housing
    masked = _mask_names(text, record).lower()
    hits = [p for p in FAIR_HOUSING_PATTERNS if re.search(p, masked)]
    # also: never echo a protected attribute that happens to be in the input
    inp = record.get("input") or {}
    for path, val in _strings(inp.get("profile") or {}):
        key = path.split(".")[0]
        if key not in ALLOWED_PROFILE_KEYS and len(val) >= 2 and re.search(rf"\b{re.escape(val.lower())}\b", low):
            violations.append(f"pii/protected value from profile.{path} appears in message")
    if hits:
        violations.append(f"fair housing: {hits}")
    else:
        states.append("fair_housing_check_passed")

    # PII
    pii_hits = []
    for name, pat in PII_PATTERNS.items():
        for m in re.finditer(pat, text):
            if name == "phone_number" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", m.group(0).strip()):
                continue
            pii_hits.append(name)
    if pii_hits and constraints.get("no_pii_leak", True):
        violations.append(f"pii: {sorted(set(pii_hits))}")
    else:
        states.append("no_pii_leak")

    # opt-out
    if constraints.get("include_opt_out_instructions", True) and not OPT_OUT_RE.search(text):
        violations.append("missing opt-out instructions")

    # channel shape
    ch = message.get("channel")
    if ch in ("sms", "voice") and subject:
        violations.append(f"{ch} must not have a subject")
    if ch == "sms":
        if len(body) > SMS_MAX_CHARS:
            violations.append(f"sms too long ({len(body)} > {SMS_MAX_CHARS})")
    if ch == "email" and not subject:
        violations.append("email needs a subject")

    # numbered reply options must be spelled out so replies can be parsed
    if ch == "sms" and decision.cta_options:
        for i, opt in enumerate(decision.cta_options, 1):
            label = decision.cta_option_labels[i - 1] if i <= len(decision.cta_option_labels) else opt
            if not re.search(rf"\b{i}\b[^.?!]*\b({re.escape(opt)}|{re.escape(label)})", body, re.I):
                violations.append(f"sms must say 'Reply {i} for {opt}' (numbered options {decision.cta_options})")
                break
        # the offered days must be described in the right week ("Thu send, Mon/Tue options" is next week)
        this_wk = re.search(r"\bthis week\b|\besta semana\b", body, re.I)
        next_wk = re.search(r"\bnext week\b|\bpr[oó]xima semana\b", body, re.I)
        if not decision.cta_same_week and this_wk:
            violations.append(f"tour options {decision.cta_options} fall next week; body says '{this_wk.group(0)}'")
        elif decision.cta_same_week and next_wk:
            violations.append(f"tour options {decision.cta_options} fall this week; body says '{next_wk.group(0)}'")
    if ch == "email" and decision.cta_link and decision.cta_link not in body:
        violations.append(f"email must include the CTA link {decision.cta_link}")

    # CTA alignment
    # greeting must use the record's name (or none) — catches names copied from the few-shot examples
    prof = (record.get("input") or {}).get("profile") or {}
    allowed = {str(n).lower() for n in (prof.get("first_name"), prof.get("preferred_name")) if n}
    m = re.match(r"\s*(hi|hello|hey|hola|dear)\s+([A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)", body, re.I)
    if m and m.group(2).lower() not in allowed | {"there"}:
        violations.append(f"greeting uses name '{m.group(2)}' which is not this contact's first name"
                          + ("" if allowed else "; record has no name, greet with 'Hi there'"))

    # only mention amenities the contact asked about (small models copy them from the examples)
    asked = " ".join(str(a) for a in (prof.get("amenity_interest") or [])).lower().replace("gym", "fitness")
    amen_text = _mask_names(text, record).lower()
    for key, pat in AMENITY_TERMS.items():
        if re.search(pat, amen_text) and key not in asked:
            violations.append(f"mentions amenity '{key}' the contact never asked about")

    if getattr(decision, "language", "en") == "es" and re.match(r"\s*(hi|hello|hey|dear)\b", body, re.I):
        violations.append("spanish message must use a Spanish greeting (Hola), not English")

    if decision.cta_type == "schedule_tour" and not re.search(r"\b(tour|visit|book|schedule|recorrido|visita|reserv|agend|cita)", low):
        violations.append("primary CTA (tour) not present in copy")

    # brand style
    if body == apply_brand_style(body) and (not subject or subject == apply_brand_style(subject)):
        states.append("brand_style_applied")
    else:
        violations.append("brand style not applied")

    return states, violations
