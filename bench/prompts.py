"""The fixed prompt sets: `python -m bench.prompts` builds them once.

Every prompt in a set is exactly the same number of tokens, so a batch needs
no padding and both backends do identical work per request. The files under
data/ are committed; the benchmark only ever reads them.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .config import REPO_ROOT

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"  # the 0.5B and 1.5B models share it
SETS = {"short": 128, "long": 1024}  # name -> prompt length in tokens

_INSTRUCTION = "Summarize the following notes in detail, then list the open questions.\n\n"
_WORDS = (
    "system model request latency batch memory token cache server client queue "
    "worker result config metric sample window budget signal error retry limit "
    "the a this each every one of in on for with from under after before while "
    "is was will can should may reads writes sends holds drops moves keeps takes "
    "fast slow large small early late fixed shared single steady idle busy cold warm"
).split()


def _filler(rng: random.Random, n_words: int) -> str:
    sentences = []
    while n_words > 0:
        size = min(rng.randint(6, 14), n_words)
        words = [rng.choice(_WORDS) for _ in range(size)]
        sentences.append(" ".join(words).capitalize() + ".")
        n_words -= size
    return " ".join(sentences)


def build(tokenizer, n_tokens: int, num_prompts: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    prompts = []
    while len(prompts) < num_prompts:
        # Overshoot, cut to the exact length, and keep the prompt only if the
        # cut text tokenizes back to that length.
        ids = tokenizer.encode(_INSTRUCTION + _filler(rng, n_tokens * 2), add_special_tokens=False)
        text = tokenizer.decode(ids[:n_tokens])
        if len(tokenizer.encode(text, add_special_tokens=False)) == n_tokens:
            prompts.append({"id": len(prompts), "n_tokens": n_tokens, "text": text})
    return prompts


def load_prompts(spec: dict) -> list[str]:
    """Return the first `num_prompts` prompts of the set named in a config."""
    path = REPO_ROOT / spec["file"]
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[: spec["num_prompts"]]
    if len(rows) < spec["num_prompts"]:
        raise ValueError(f"{path} has {len(rows)} prompts, config asks for {spec['num_prompts']}")
    wrong = {r["n_tokens"] for r in rows} - {spec["input_tokens"]}
    if wrong:
        raise ValueError(f"{path} holds {sorted(wrong)}-token prompts, config says {spec['input_tokens']}")
    return [r["text"] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the fixed prompt sets under data/.")
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="replace existing prompt files")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    for name, n_tokens in SETS.items():
        path = REPO_ROOT / "data" / f"prompts_{name}.jsonl"
        if path.exists() and not args.force:
            print(f"{path} exists; the prompt set is meant to stay fixed (use --force to replace it)")
            continue
        path.parent.mkdir(exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in build(tokenizer, n_tokens, args.num_prompts, args.seed):
                f.write(json.dumps(row) + "\n")
        print(f"wrote {args.num_prompts} prompts of {n_tokens} tokens to {path}")


if __name__ == "__main__":
    main()
