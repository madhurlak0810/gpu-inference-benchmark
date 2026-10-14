"""The fixed prompt sets: `python -m bench.prompts` builds them once.

Every prompt in a set is exactly the same number of tokens, so a batch needs
no padding and both backends do identical work per request. The files under
data/ are committed; the benchmark only ever reads them.

The shared-prefix set is different: one fixed prefix plus a suffix drawn fresh
for every request, so a prefix cache can reuse the prefix and nothing else.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .config import REPO_ROOT

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"  # the 0.5B and 1.5B models share it
SETS = {"short": 128, "long": 1024}  # name -> prompt length in tokens
SHARED_PREFIX_TOKENS = 512
SHARED_PREFIX_FILE = "data/prompts_shared_prefix.json"

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


def build_shared_prefix(tokenizer, prefix_tokens: int, seed: int) -> dict:
    """A prefix of exactly `prefix_tokens` tokens and the words that extend it one token each.

    With those, a prompt of any exact length can be put together at run time
    without a tokenizer: the prefix, then one " word" per remaining token.
    """
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    rng = random.Random(seed)
    words = [w for w in sorted(set(_WORDS)) if len(encode(" " + w)) == 1]
    while True:
        ids = encode(_INSTRUCTION + _filler(rng, prefix_tokens * 2))[:prefix_tokens]
        prefix = tokenizer.decode(ids)
        # Keep the prefix only if it tokenizes back to itself, alone and with
        # a suffix after it: the suffix must not merge into its last token.
        samples = [_suffix(rng, words, n) for n in (1, 16, 512)]
        if encode(prefix) == ids and all(
            encode(prefix + s)[:prefix_tokens] == ids and len(encode(prefix + s)) == prefix_tokens + n
            for s, n in zip(samples, (1, 16, 512))
        ):
            return {"prefix_tokens": prefix_tokens, "prefix": prefix, "words": words}


def _suffix(rng: random.Random, words: list[str], n_tokens: int) -> str:
    return "".join(" " + rng.choice(words) for _ in range(n_tokens))


def prompt_source(spec: dict, seed: int):
    """Return `take(n)`, giving the next n prompts as (prompt id, text) pairs.

    A fixed set starts again from its first prompt on every call, so each
    setting and repeat sends the same prompts. A set with
    `shared_prefix_tokens` never repeats a prompt: only the prefix recurs.
    """
    if "shared_prefix_tokens" not in spec:
        prompts = load_prompts(spec)
        return lambda n: [(i % len(prompts), prompts[i % len(prompts)]) for i in range(n)]

    data = json.loads((REPO_ROOT / spec["file"]).read_text(encoding="utf-8"))
    if data["prefix_tokens"] != spec["shared_prefix_tokens"]:
        raise ValueError(f"{spec['file']} has a {data['prefix_tokens']}-token prefix, "
                         f"config says {spec['shared_prefix_tokens']}")
    rng = random.Random(seed)
    suffix_tokens = spec["input_tokens"] - data["prefix_tokens"]
    sent = 0

    def take(n: int) -> list[tuple[int, str]]:
        nonlocal sent
        out = [(sent + i, data["prefix"] + _suffix(rng, data["words"], suffix_tokens)) for i in range(n)]
        sent += n
        return out

    return take


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

    path = REPO_ROOT / SHARED_PREFIX_FILE
    if path.exists() and not args.force:
        print(f"{path} exists; the prompt set is meant to stay fixed (use --force to replace it)")
    else:
        data = build_shared_prefix(tokenizer, SHARED_PREFIX_TOKENS, args.seed)
        path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote a {SHARED_PREFIX_TOKENS}-token shared prefix and {len(data['words'])} suffix words to {path}")


if __name__ == "__main__":
    main()
