# Context-aware messaging agent

Reads a contact record (profile, consent, preferences, context, constraints) and decides **whether** to
message, **how** (channel + time), **what** to say, and **what happens next**. Output is shaped exactly like
the `expected` block in `sample.jsonl`, plus a rationale, the compliance states it passed, and a trace.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# offline: deterministic policy + templates, scored against `expected`
.venv/bin/python run.py data/sample.jsonl

# free local LLM (Ollama, no key): one-time `ollama pull qwen2.5:7b`
.venv/bin/python run.py data/holdout.jsonl --llm --examples data/sample.jsonl --out outputs/holdout.jsonl

# Claude instead (needs ANTHROPIC_API_KEY)
.venv/bin/python run.py data/holdout.jsonl --llm claude --examples data/sample.jsonl --out outputs/holdout.jsonl

.venv/bin/python tests/test_agent.py
```

`run.py` prints one JSON object per case (copy-paste ready), writes all of them to `--out`, and if a case
has `expected` it prints a scorecard. `--full` adds the decision trace.

## Architecture

```
record ──► policy.decide()  ──► composer (Claude few-shot │ template) ──► enforce ──► guardrails.check()
             hard rules           learns tone/structure        policy wins   ├─ pass ──► output
             consent, channel,    from nearest examples                      └─ fail ──► retry with feedback
             send time, CTA,                                                              └─► template fallback
             opt-out, cadence
```

| Layer | File | Why it is separate |
|---|---|---|
| Policy | `agent/policy.py` | Consent, channel, quiet hours, send-time math, and opt-out wording are compliance decisions. They have to be right every time, so they're code, not prompts. |
| Composer | `agent/composer.py` | Copy is where "semantic match" applies. The LLM gets the most similar training cases as few-shot examples plus the policy output as constraints, and returns a strict JSON schema. There are two interchangeable backends with the same prompt: **Ollama** (local `qwen2.5:7b`, free, uses Ollama's JSON-schema `format`) and **Claude** (structured outputs). The LLM writes only the copy. Its proposed `next_action` is logged, but the policy decides it. |
| Guardrails | `agent/guardrails.py` | Every message is checked for consent, Fair Housing language, PII leaks (regex plus non-allowlisted profile values echoed back), opt-out text, channel shape (SMS ≤320 chars with no subject; email needs a subject), CTA presence (numbered SMS options, email link), and brand style. Two checks catch details the model copies from the examples: a greeting with a name that isn't this contact's, and amenities the contact never asked about. A failure triggers a retry with the violations as feedback, then the template as a last resort. |
| Orchestrator | `agent/agent.py` | Ties the layers together and measures latency. It also handles inbound replies. Opt-outs, including natural phrasing like "please stop texting me" but not "stop by", suppress marketing and send one SMS confirmation. "Not interested" closes the lead with no message. A picked option ("1") gets a tour confirmation, and anything else is acknowledged and handed to a leasing agent. |
| Robustness | `run.py`, `policy.py` | A record that can't be processed becomes `next_action: manual_review` with nothing sent; it never stops the batch. Inputs are parsed defensively: timezone abbreviations, naive or epoch timestamps, unknown or string channels, and non-string fields. |
| Eval | `agent/evaluate.py` | Decision fields are compared exactly: send/no-send, channel, send_at, subject present, CTA type, next_action. Prose is compared semantically: content-word F1, must-carry facts (name, options or link, STOP), and personalization. |

## Rules learned from the data

From the two samples (kept explicit so they're easy to challenge):

- **Channel** is the first entry in `channel_preferences` whose consent flag is true (sample 2 prefers SMS but only email is opted in, so it gets email).
- **Send time** is `last_interaction` (converted to the contact's timezone) plus N days, where N comes from `dayN` in task_id (minimum 1). The local hour is SMS 09:00, email 10:00, voice 11:00, always inside 08:00–21:00.
- **CTA**: `book_tour` maps to `schedule_tour`. SMS offers two numbered weekday options at least 2 days after the send day. Email uses the link `https://<property>.example/tour`.
- **Cadence**: lifecycle `new` starts `{persona}_welcome_{short|long}_horizon`. Otherwise `follow_up_in_days: 3`. Horizon is short when the move date is within 45 days.
- **No send** happens when no channel has consent, the contact replied STOP or is flagged do-not-contact, or the lifecycle stage is terminal. The output is `next_message: null` with `next_action: {type: suppress, reason}`.

Two examples can't pin these rules down. The voice hour, the 45-day horizon cutoff and the follow-up interval are guesses. The LLM path exists so the hold-out examples can override them where the deterministic rules have no evidence.

## Talking points / trade-offs

- **Why hybrid instead of pure LLM?** Compliance (TCPA consent and quiet hours, Fair Housing, CAN-SPAM opt-out) must never be wrong, while copy only needs to be close. Hard rules go in code and tone goes in the model, and every draft is checked before it ships. That keeps `safety_violations_max: 0` on every case tested. It is not a proof: the checks are pattern-based. Fair Housing checks match steering phrases, not bare words, and mask the contact and property names, so "White Oak Apartments" or a contact named Christian doesn't trip them. They can still miss creative wording.
- **Latency (`p95_latency_ms: 2000`)**: measured on this laptop, the offline path takes about 2 ms per case, while local qwen2.5:7b takes 10–25 s (it often needs one repair retry). Neither LLM path meets 2 s. Production options: a hosted small model (`--model claude-haiku-5-5`), caching the system prompt and examples, or pre-generating copy per segment offline and only filling slots at send time. The rules layer already meets the target.
- **Small-model failure modes seen in testing** (all caught by guardrails, then fixed by a retry or the template): it copied "Taylor" into a record with no name, mentioned a pool and gym the contact never asked about, left out the numbered SMS reply options, and invented a `next_action` of `opt_out`. This is why decisions stay in code and every draft is checked.
- **Learning from data only**: example retrieval uses Jaccard similarity over record features (persona, stage, consent pattern, keys present, task_id tokens). As the training set grows, swap in embeddings and add a held-out eval to tune the prompt.
- **Evaluation**: exact match on decision fields and semantic match on prose. An LLM-as-judge for "does this body mean the same thing" is the next step. Token F1 penalizes valid paraphrases.
- **Scope limits**: non-tour CTAs (apply, renew, pay rent, maintenance) and resident personas get plain generic copy, because the samples show only prospects and tours. Languages other than English and Spanish fall back to English. A record with no `last_interaction` schedules relative to the current time.
- **Known overfit**: the templates reproduce the two sample messages almost word for word. That makes them a safe fallback, not proof the agent generalizes. `data/edge_cases.jsonl` covers no consent, STOP replies, Spanish, voice only, protected attributes in the profile, and an empty record.
