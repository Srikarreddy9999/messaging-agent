#!/usr/bin/env python3
"""CLI.

  # offline (deterministic policy + templates), score against expected
  python run.py data/sample.jsonl

  # with a free local LLM (Ollama) composing the copy, few-shot from the training file
  python run.py data/holdout.jsonl --llm --examples data/sample.jsonl --out outputs/holdout_outputs.jsonl

  # same, with OpenAI or Claude (needs OPENAI_API_KEY / ANTHROPIC_API_KEY)
  python run.py data/holdout.jsonl --llm openai --examples data/sample.jsonl
  python run.py data/holdout.jsonl --llm claude --examples data/sample.jsonl

Hold-out records without `expected` are just answered; records with `expected` are also scored.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent.agent import MessagingAgent, classify_reply  # noqa: E402
from agent.evaluate import batch_thresholds, p95, reply_f1, score  # noqa: E402

ANSWER_KEYS = ("task_id", "next_message", "next_action", "rationale", "states")
LLM_SOURCES = ("llm(",)


def load_jsonl(path: str) -> list:
    """Every non-blank line becomes one item, so outputs always line up 1:1 with inputs.
    A line that isn't valid JSON is kept as an error marker and answered with manual_review."""
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"line {i} is not valid JSON ({e}); it will be answered with manual_review", file=sys.stderr)
            rows.append({"__invalid_line__": i, "task_id": f"invalid_line_{i}"})
    return rows


def error_output(r, e) -> dict:
    return {"task_id": r.get("task_id") if isinstance(r, dict) else None, "next_message": None,
            "next_action": {"type": "manual_review", "reason": f"agent error: {e}"},
            "rationale": "Could not process this record safely, so nothing is sent.",
            "states": [], "violations": [], "meta": {"source": "error", "latency_ms": 0.0, "trace": [str(e)]}}


def main() -> None:
    # Windows consoles default to cp1252; message copy has em dashes and curly quotes.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL of test cases")
    ap.add_argument("--examples", default=None, help="training JSONL for few-shot (default: the input file itself, leave-one-out)")
    ap.add_argument("--llm", nargs="?", const="ollama", choices=["ollama", "openai", "claude"], default=None,
                    help="LLM backend: ollama (free, local; default when flag given), openai (OPENAI_API_KEY) or claude (ANTHROPIC_API_KEY)")
    ap.add_argument("--model", default=None, help="model id (defaults: ollama qwen2.5:7b, openai gpt-5-mini, claude claude-opus-5-5)")
    ap.add_argument("--out", default="outputs/outputs.jsonl", help="full output incl. trace; answers-only copy goes to <out>.answers.jsonl")
    ap.add_argument("--full", action="store_true", help="print full output incl. trace")
    ap.add_argument("--decisions", choices=["llm", "rules"], default="llm",
                    help="who decides channel/send time/next action: the LLM from the examples (within hard limits) or the default rules")
    ap.add_argument("--judge", choices=["ollama", "openai", "claude"], default=None,
                    help="LLM judge for message meaning when records have `expected`")
    ap.add_argument("--reply-labels", default=str(Path(__file__).parent / "data" / "reply_labels.jsonl"),
                    help="labelled replies for reply_classification_f1 ('' to skip)")
    ap.add_argument("--require-llm", action="store_true",
                    help="exit 1 unless every sent message was written by the LLM (proves the integration works)")
    args = ap.parse_args()

    records = load_jsonl(args.input)
    examples = [r for r in (load_jsonl(args.examples) if args.examples else records) if isinstance(r, dict)]
    agent = MessagingAgent(examples=examples, llm=args.llm, model=args.model, decisions=args.decisions)
    judge = None
    if args.judge:
        from agent.judge import make_judge
        judge = make_judge(args.judge)

    outputs, scores = [], []
    out_path = Path(args.out)
    answers_path = out_path.with_name(out_path.stem + ".answers.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f, open(answers_path, "w", encoding="utf-8") as fa:
        for r in records:
            # One bad record must never cost the other outputs.
            try:
                if not isinstance(r, dict) or "__invalid_line__" in r:
                    raise ValueError("record is not a JSON object")
                out = agent.run(r)
            except Exception as e:  # noqa: BLE001
                out = error_output(r, repr(e))
            outputs.append(out)
            answer = {k: out.get(k) for k in ANSWER_KEYS}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            fa.write(json.dumps(answer, ensure_ascii=False) + "\n")
            f.flush()
            fa.flush()
            print(json.dumps(out if args.full else answer, ensure_ascii=False, indent=2), flush=True)
            if isinstance(r, dict) and "expected" in r:
                scores.append(score(r, out, judge))

    # --- export check: exactly one valid JSON object per input record --------------------------
    lines = [l for l in answers_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    valid = 0
    for l in lines:
        try:
            o = json.loads(l)
            valid += all(k in o for k in ANSWER_KEYS)
        except json.JSONDecodeError:
            pass
    ok_export = valid == len(lines) == len(records)
    print(f"\n{len(records)} input records -> {valid}/{len(lines)} valid outputs "
          f"{'OK' if ok_export else 'MISMATCH'}", file=sys.stderr)
    print(f"answers (copy/paste this file): {answers_path}", file=sys.stderr)
    print(f"full output with traces:        {out_path}", file=sys.stderr)

    lat = [o["meta"]["latency_ms"] for o in outputs]
    if lat:
        print(f"latency ms: median {statistics.median(lat)}, p95 {p95(lat)}", file=sys.stderr)
    src = {}
    for o in outputs:
        src[o["meta"].get("source", "?")] = src.get(o["meta"].get("source", "?"), 0) + 1
    print(f"message source per case: {src}", file=sys.stderr)

    reply_eval = None
    if args.reply_labels and Path(args.reply_labels).exists():
        reply_eval = reply_f1(args.reply_labels, classify_reply)
    if scores:
        print("\nscorecard:", file=sys.stderr)
        for s in scores:
            print("  " + json.dumps(s, ensure_ascii=False), file=sys.stderr)
        print(f"pass rate: {sum(s['pass'] for s in scores)}/{len(scores)}", file=sys.stderr)
        if not judge:
            print("note: message meaning was not judged (token F1 is only a proxy); add --judge <backend>", file=sys.stderr)
    batch = batch_thresholds([r for r in records if isinstance(r, dict)], outputs, reply_eval)
    if batch:
        print("\nrun-level thresholds:", file=sys.stderr)
        for b in batch:
            status = "PASS" if b["ok"] else ("n/a" if b["ok"] is None else "FAIL")
            print(f"  {status}  {b['threshold']}: value {b['value']} vs limit {b['limit']}"
                  + (f" ({b['n_labelled']} labelled replies, hand-written, not provided by the task)" if "n_labelled" in b else "")
                  + (f" ({b['note']})" if "note" in b else ""), file=sys.stderr)
        if reply_eval and reply_eval["errors"]:
            print(f"  reply classifier errors: {reply_eval['errors']}", file=sys.stderr)

    exit_code = 0 if ok_export else 2
    if args.require_llm:
        sent = [o for o in outputs if o.get("next_message") is not None and o["meta"].get("source") not in ("reply_handler", "policy")]
        not_llm = [o for o in sent if not str(o["meta"].get("source", "")).startswith(LLM_SOURCES)]
        if not args.llm:
            print("\nREQUIRE-LLM: FAIL (no --llm backend given)", file=sys.stderr)
            exit_code = 1
        elif not_llm:
            print(f"\nREQUIRE-LLM: FAIL - {len(not_llm)}/{len(sent)} sent messages did not come from the LLM:", file=sys.stderr)
            for o in not_llm:
                why = [t for t in o["meta"]["trace"] if "llm" in t or "violation" in t or "blocked" in t]
                print(f"  {o['task_id']}: source={o['meta'].get('source')} | {why[-3:]}", file=sys.stderr)
            exit_code = 1
        else:
            print(f"\nREQUIRE-LLM: PASS - all {len(sent)} sent messages were written by the LLM", file=sys.stderr)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
