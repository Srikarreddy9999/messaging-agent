#!/usr/bin/env python3
"""CLI.

  # offline (deterministic policy + templates), score against expected
  python run.py data/sample.jsonl

  # with a free local LLM (Ollama) composing the copy, few-shot from the training file
  python run.py data/holdout.jsonl --llm --examples data/sample.jsonl --out outputs/holdout_outputs.jsonl

  # same, with Claude
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
from agent.agent import MessagingAgent  # noqa: E402
from agent.evaluate import score  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"skipping line {i}: {e}", file=sys.stderr)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL of test cases")
    ap.add_argument("--examples", default=None, help="training JSONL for few-shot (default: the input file itself, leave-one-out)")
    ap.add_argument("--llm", nargs="?", const="ollama", choices=["ollama", "claude"], default=None,
                    help="LLM for copy: ollama (free, local; default when flag given) or claude (needs Anthropic key)")
    ap.add_argument("--model", default=None, help="model id (ollama default qwen2.5:7b, claude default claude-opus-5-5)")
    ap.add_argument("--out", default="outputs/outputs.jsonl")
    ap.add_argument("--full", action="store_true", help="print full output incl. trace")
    args = ap.parse_args()

    records = load_jsonl(args.input)
    examples = load_jsonl(args.examples) if args.examples else records
    agent = MessagingAgent(examples=examples, llm=args.llm, model=args.model)

    outputs, scores = [], []
    for r in records:
        out = agent.run(r)
        outputs.append(out)
        if "expected" in r:
            scores.append(score(r, out))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # Copy-paste friendly: the answer fields only, one per case
    for o in outputs:
        view = o if args.full else {k: o[k] for k in ("task_id", "next_message", "next_action", "rationale", "states")}
        print(json.dumps(view, ensure_ascii=False, indent=2))

    lat = [o["meta"]["latency_ms"] for o in outputs]
    print(f"\n{len(outputs)} cases -> {args.out}", file=sys.stderr)
    if lat:
        p95 = sorted(lat)[max(0, int(round(0.95 * len(lat))) - 1)]
        print(f"latency ms: median {statistics.median(lat)}, p95 {p95}", file=sys.stderr)
    if scores:
        print("\nscorecard:", file=sys.stderr)
        for s in scores:
            print("  " + json.dumps(s), file=sys.stderr)
        print(f"pass rate: {sum(s['pass'] for s in scores)}/{len(scores)}", file=sys.stderr)


if __name__ == "__main__":
    main()
