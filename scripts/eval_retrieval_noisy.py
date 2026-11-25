"""D1 x D2: retrieval with *noisy* pantry surface forms, with and without
ingredient normalization.

Each hidden-recipe query ingredient is perturbed the way users type: regional
synonym ("eggplant" -> "brinjal"), quantities/prep words ("2 cups chopped ..."),
plurals, and occasional typos. We then retrieve either with the raw strings or
after D1 normalization.

    python scripts/eval_retrieval_noisy.py --n 500
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pantryrag import corpus  # noqa: E402
from pantryrag.lexicon import SYNONYMS  # noqa: E402
from pantryrag.normalize import Normalizer, canonical_form  # noqa: E402
from pantryrag.pantry_parser import PantryQuery  # noqa: E402
from pantryrag.retrieval import Retriever  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_retrieval import MODES, evaluate, make_queries  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PREFIX = ["2 cups", "1 tbsp", "some", "fresh", "a handful of", "200g", "leftover", "chopped", ""]
SUFFIX = [", diced", ", sliced", " (optional)", "", "", ""]


def perturb(ing: str, reverse: dict, rng: random.Random) -> str:
    """Make a canonical ingredient look user-typed: regional synonym or plural,
    sometimes a typo, sometimes a quantity/prep prefix and suffix."""
    s = ing
    if ing in reverse and rng.random() < 0.7:
        s = rng.choice(reverse[ing])
    elif rng.random() < 0.5 and not s.endswith("s"):
        s = s + ("es" if s.endswith(("o", "ch", "sh")) else "s")
    # typo: swap two adjacent characters
    if len(s) > 5 and rng.random() < 0.15:
        i = rng.randint(1, len(s) - 3)
        s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if rng.random() < 0.5:
        s = f"{rng.choice(PREFIX)} {s}{rng.choice(SUFFIX)}".strip()
    return s


def main():
    """Evaluate every retrieval mode on clean, noisy-raw and noisy-normalised queries."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    recipes, vocab = corpus.load()
    R = Retriever.load(recipes)
    norm = Normalizer([k for k, c in vocab.items() if c >= 2], vocab)
    # canonical name -> regional names that map to it, used to inject synonyms
    reverse = defaultdict(list)
    for k, v in SYNONYMS.items():
        reverse[canonical_form(v)].append(k)
    rng = random.Random(args.seed)
    base = make_queries(recipes, args.n, rng)
    noisy_raw, noisy_norm = [], []
    for rid, q in base:
        raw = [perturb(i, reverse, rng) for i in q.ingredients]
        noisy_raw.append((rid, PantryQuery("", ingredients=[r.lower() for r in raw])))
        noisy_norm.append((rid, PantryQuery("", ingredients=norm.normalize_many(raw))))
    examples = [(q1.ingredients, q2.ingredients) for (_, q1), (_, q2) in zip(noisy_raw[:5], noisy_norm[:5])]
    out = {"n": args.n, "corpus_source": (corpus.PROCESSED / "SOURCE").read_text().strip(),
           "examples": examples, "clean": {}, "noisy_raw": {}, "noisy_normalized": {}}
    for mode in MODES:
        for name, qs in (("clean", base), ("noisy_raw", noisy_raw), ("noisy_normalized", noisy_norm)):
            out[name][mode] = {k: round(v, 3) for k, v in evaluate(R, qs, mode).items()}
            print(name, mode, out[name][mode], flush=True)
    (ROOT / "results/retrieval_noisy.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
