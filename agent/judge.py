"""LLM-as-judge for message meaning. Token overlap can't tell a valid paraphrase from a wrong message;
this asks a model whether the agent's message would do the same job as the expected one."""
from __future__ import annotations

import json

JUDGE_SYSTEM = """You grade an automated leasing message against a reference message for the same contact.
They are EQUIVALENT if a reasonable leasing manager would accept either: same intent and call to action,
same key facts (name, property, offered days or link, stated interests), opt-out present, no extra claims
that the reference lacks. Wording, order and tone may differ. Be strict about wrong facts or a missing CTA."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"equivalent": {"type": "boolean"}, "score": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["equivalent", "score", "reason"],
    "additionalProperties": False,
}


def make_judge(backend: str, model: str | None = None):
    from .composer import ClaudeComposer, OllamaComposer, OpenAIComposer

    client = {"ollama": OllamaComposer, "openai": OpenAIComposer, "claude": ClaudeComposer}[backend](model=model)

    def judge(record: dict, expected_msg: dict, actual_msg: dict) -> dict:
        user = (f"<contact>{json.dumps(record.get('input'), ensure_ascii=False)}</contact>\n"
                f"<reference>{json.dumps(expected_msg, ensure_ascii=False)}</reference>\n"
                f"<candidate>{json.dumps(actual_msg, ensure_ascii=False)}</candidate>")
        try:
            return client.json_call(JUDGE_SYSTEM, user, JUDGE_SCHEMA)
        except Exception as e:  # noqa: BLE001
            return {"equivalent": None, "score": None, "reason": f"judge error: {e!r}"}

    return judge
