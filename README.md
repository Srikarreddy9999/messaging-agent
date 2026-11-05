# Context-aware messaging agent

Reads a contact record (profile, consent, preferences, context, constraints) and decides **whether** to
message, **how** (channel + time), **what** to say, and **what happens next**. Output is shaped exactly like
the `expected` block in `sample.jsonl`, plus a rationale, the compliance states it passed, and a trace that
says where every decision came from.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# offline: defaults + templates, scored against `expected`
.venv/bin/python run.py data/sample.jsonl

# with an LLM (pick one). The examples file is the training data the LLM learns from.
.venv/bin/python run.py data/holdout.jsonl --llm openai --examples data/sample.jsonl --out outputs/holdout.jsonl
.venv/bin/python run.py data/holdout.jsonl --llm          --examples data/sample.jsonl --out outputs/holdout.jsonl  # Ollama, free
.venv/bin/python run.py data/holdout.jsonl --llm claude   --examples data/sample.jsonl --out outputs/holdout.jsonl

.venv/bin/python tests/test_agent.py
```

**Windows (PowerShell)**: use `.venv\Scripts\python`, and always run `run.py` with a data file.
Don't run `agent\agent.py` directly: it's a package module, so that gives "attempted relative import".

```powershell
py -m venv .venv; .venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run.py data\sample.jsonl
$env:OPENAI_API_KEY = "sk-..."   # current terminal only
.venv\Scripts\python run.py data\holdout.jsonl --llm openai --examples data\sample.jsonl --out outputs\holdout.jsonl
```

A ChatGPT subscription does not include API access. The API is billed separately and prepaid; a few dollars
of credit covers this many times over.

### Options

| Flag | What it does |
|---|---|
| `--llm ollama\|openai\|claude` | Backend. Without it, the agent runs offline on defaults and templates. |
| `--decisions llm\|rules` | `llm` (default): the LLM proposes channel, send time and next action from the examples, and code accepts each one only if it passes the hard limits. `rules`: the LLM writes copy only, and decisions come from the labelled defaults. |
| `--require-llm` | Exits 1 unless every sent message was actually written by the LLM. A template fallback can look fine, so this is how you prove the integration worked. |
| `--judge ollama\|openai\|claude` | An LLM judge checks whether each message means the same as `expected`. Without it, meaning is not judged; token F1 is reported only as a proxy. |
| `--reply-labels FILE` | Labelled replies used for `reply_classification_f1_min`. Defaults to `data/reply_labels.jsonl`. |
| `--full` | Prints the full output, including the trace. |

### Output files

`--out outputs/holdout.jsonl` writes two files:
- **`outputs/holdout.answers.jsonl`**: one line per input record with only the answer fields. **This is the file to copy and paste.**
- **`outputs/holdout.jsonl`**: the same answers plus `meta`: the source of each message, which decisions the LLM made versus the defaults, latency, and the full trace.

At the end of every run it prints `N input records -> N/N valid outputs OK`. Invalid JSON lines and failed records still get an output line (`manual_review`), so the count always matches.

## Architecture

```
record ─► decide(): hard limits + labelled defaults
            │  suppress? (no consent, opt-out, terminal stage) ─► next_message null
            ▼
          LLM (few-shot from nearest examples) proposes: send? channel, send_at, copy, next_action
            ▼
          _accept(): each decision checked against the hard limits; a failing field falls back to its default
            ▼
          guardrails.check(): fail ─► retry with feedback ─► template ─► still failing ─► manual_review, nothing sent
```

| Layer | File | Role |
|---|---|---|
| Decisions | `agent/policy.py` | **Hard limits** are never overridden: consent per channel, opt-out and do-not-contact, terminal lifecycle stages, quiet hours 08:00–21:00 local, a send time after the last interaction, and opt-out wording. **Defaults** are labelled assumptions (see below), used only when there's no LLM, the LLM's proposal breaks a limit, or `--decisions rules` is set. |
| Composer | `agent/composer.py` | Same prompt for every backend: the nearest examples, plus the hard limits, the defaults (with their basis) and the CTA for each channel. Strict JSON-schema output on Ollama, OpenAI and Claude. |
| Guardrails | `agent/guardrails.py` | Checks every draft: consent, Fair Housing steering phrases (contact and property names are masked first), PII, opt-out text, channel shape, CTA presence, brand style, plus names and amenities copied from the examples. |
| Orchestrator | `agent/agent.py` | Accepts or rejects LLM decisions one field at a time and records each source in `meta.decided_by`. It also handles inbound replies: an opt-out suppresses marketing and sends one confirmation, "not interested" closes the lead, a picked option confirms the tour, and anything else is handed to a leasing agent. **Fail-safe:** if every draft fails validation, nothing is sent and the action is `manual_review`. |
| Eval | `agent/evaluate.py`, `agent/judge.py` | A per-case `pass` requires every supplied threshold (`personalization_score_min`, `safety_violations_max`, required states) plus the decision fields, and semantic equivalence when `--judge` is used. Run-level thresholds (`p95_latency_ms`, `reply_classification_f1_min`) print as PASS/FAIL. |

## Assumptions (not learned rules)

Two examples can't establish any of these. They live in `ASSUMPTIONS` in `agent/policy.py`, and the trace names each one when it's used.

| Assumption | Value | Basis |
|---|---|---|
| Default channel | first consented entry in `channel_preferences` | consistent with both samples (n=2) |
| Send hour | SMS 09:00, email 10:00, voice 11:00 local | SMS and email each match one sample; voice is a guess |
| Send day | `dayN` in task_id, else the next day | task_id matched the offset in both samples; could be naming coincidence |
| Short vs long horizon | move within 45 days | guess between the samples' 33 and 71 days |
| Follow-up | 3 days | one sample |
| LLM send-time bound | at most 14 days after the last interaction | sanity limit; guess |

With `--decisions llm`, the LLM can depart from these whenever the examples show a different pattern, as long as it stays inside the hard limits.

## What was measured

All numbers below come from this laptop.

- **Samples, offline:** 2/2 pass. The defaults were fitted to these two cases, so this proves little.
- **Samples, Ollama qwen2.5:7b, leave-one-out:** with `--decisions llm`, 1/2 pass. With only one example to learn from (an email case), the LLM chose email for the SMS case. That's a legal choice, so the checks accepted it. This is the real cost of learning decisions from tiny data. With `--decisions rules`, 2/2 pass.
- **12-record rehearsal (`data/holdout_rehearsal.jsonl`, no `expected`), Ollama:** 12/12 valid outputs, and `--require-llm` passed (all 8 sent messages were written by the LLM). p95 latency was about 22 s, which **fails** the 2000 ms threshold. The offline path takes about 3 ms.
- **Reply classifier:** macro F1 was 0.921 on 42 replies I labelled myself (`data/reply_labels.jsonl`) before measuring. After fixing a real rule-order bug it found ("reschedule my tour" was classified as a tour request), it scores 0.975 on the same set. That second number is optimistic, because the set is small, hand-written, and was used for the fix.
- **OpenAI path:** tested only against a local fake server, which checked the request shape and the fallback. It has not been run against OpenAI itself. Run it with `--require-llm` and a real key before relying on it.

## Talking points / trade-offs

- **Who decides what:** safety lives in code because it must never be wrong. Workflow decisions come from the examples through the LLM, and code only checks they're legal. Tone and wording come from the LLM. The trade-off is the leave-one-out result above: with tiny data, LLM decisions are less reliable than the defaults, which is why `--decisions rules` exists.
- **Latency:** neither LLM path meets 2 s per case on this setup. Options: a smaller hosted model, prompt caching, or pre-generating copy per segment and only filling slots at send time.
- **Evaluation honesty:** decision fields are compared exactly; message meaning needs `--judge`. Token F1 is only a proxy, and personalization is our own approximation, since the task doesn't define it.
- **Scope limits:** non-tour CTAs and resident personas get plain generic template copy, because the samples show only prospects booking tours. Languages other than English and Spanish fall back to English. A record with no `last_interaction` is scheduled relative to the current time.
