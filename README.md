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
.venv\Scripts\python run.py data\holdout.jsonl --llm openai --decisions rules --examples data\sample.jsonl --out outputs\holdout.jsonl
```

Use `--decisions rules` for the hold-out. With only two examples, the LLM picks channel and time less reliably than the defaults (see "What was measured").

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

## Edge cases

`data/edge_cases.jsonl` has 34 records, and `test_edge_cases_file` asserts the expected behaviour for each one:

| Area | Cases | Expected behaviour |
|---|---|---|
| Suppression | no consent, `do_not_contact`, `closed_lost`, consent given as strings ("yes"), empty record | `next_message: null`, with the reason |
| Replies | STOP, Spanish "cancelar", "not interested", "1", reschedule, general question, "can I stop by" | opt-out confirmation, `close_lead`, `confirm_tour`, or a handoff to an agent. "Stop by" is **not** an opt-out |
| Timing | late-night last contact, Thursday and Friday sends, epoch timestamp, no `last_interaction`, invalid timezone | always 08:00–21:00 local. Tour days are named in the right week, and both days fall in the same week |
| Channel | SMS preferred but only email consented, empty preferences, voice only, Spanish voice | first **consented** channel |
| Content | preferred name, Unicode name with a long property name, unknown amenity, PII and protected classes in the profile, unsupported language (fr) | preferred name used, no PII or protected attributes in the copy, falls back to English |
| Other CTAs | resident pay rent (SMS), maintenance (email), Spanish resident variants | generic copy with a link or reply prompt, Spanish when `language` is `es` |
| Bad input | a malformed JSON line | still gets an output line, with `manual_review` |

## What was measured

All numbers below come from this laptop.

- **Samples, offline:** 2/2 pass. The defaults were fitted to these two cases, so this proves little.
- **Samples, Ollama qwen2.5:7b, leave-one-out:** with `--decisions llm`, 1/2 pass. With only one example to learn from (an email case), the LLM chose email for the SMS case. That's a legal choice, so the checks accepted it. This is the real cost of learning decisions from tiny data. With `--decisions rules`, 2/2 pass.
- **12-record rehearsal (`data/holdout_rehearsal.jsonl`, no `expected`), Ollama:** 12/12 valid outputs, and `--require-llm` passed (all 8 sent messages were written by the LLM). p95 latency was about 22 s, which **fails** the 2000 ms threshold. The offline path takes about 3 ms.
- **Reply classifier:** macro F1 was 0.921 on 42 replies I labelled myself (`data/reply_labels.jsonl`) before measuring. After fixing a real rule-order bug it found ("reschedule my tour" was classified as a tour request), it scores 0.975 on the same set. That second number is optimistic, because the set is small, hand-written, and was used for the fix.
- **OpenAI (gpt-5-mini), samples, leave-one-out, judged by an OpenAI LLM judge:** with `--decisions llm`, 1/2 pass. It made the same email-for-SMS choice as Ollama. With `--decisions rules`, **2/2 pass**, and the judge rated both messages semantically equivalent to `expected`. p95 latency was about 9.6 s.
- **12-record rehearsal, OpenAI, `--decisions rules`:** 12/12 valid outputs, and `--require-llm` passed (all 8 sent messages were written by the LLM on the first attempt). p95 latency was about 10 s, which **fails** the 2000 ms threshold.
- **Bugs that rehearsal found, now fixed:**
  - SMS tour messages said "this week" while the offered days fell next week. A Thursday send offering Mon/Tue is one example. The phrase is now corrected automatically, and a guardrail rejects the wrong week.
  - The Spanish move timing came out in English ("a mid‑January"). It's now localized ("mediados de enero").
  - Spanish messages must now open with a Spanish greeting.
  - The tour-CTA check now accepts "agendar" and "cita".

## Talking points / trade-offs

- **Who decides what:** safety lives in code because it must never be wrong. Workflow decisions come from the examples through the LLM, and code only checks they're legal. Tone and wording come from the LLM. The trade-off is the leave-one-out result above: with tiny data, LLM decisions are less reliable than the defaults, which is why `--decisions rules` exists.
- **Latency:** neither LLM path meets 2 s per case on this setup. Options: a smaller hosted model, prompt caching, or pre-generating copy per segment and only filling slots at send time.
- **Evaluation honesty:** decision fields are compared exactly; message meaning needs `--judge`. Token F1 is only a proxy, and personalization is our own approximation, since the task doesn't define it.
- **Scope limits:** non-tour CTAs and resident personas get plain generic template copy, because the samples show only prospects booking tours. Languages other than English and Spanish fall back to English. A record with no `last_interaction` is scheduled relative to the current time.
