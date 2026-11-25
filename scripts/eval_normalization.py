"""D1 evaluation: ingredient normalization accuracy (with ablation) and
pantry-query parsing accuracy on hand-labelled samples.

    python scripts/eval_normalization.py
Writes results/normalization.json and results/normalization_errors.csv.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pantryrag import corpus  # noqa: E402
from pantryrag.normalize import Normalizer, canonical_form  # noqa: E402
from pantryrag.pantry_parser import parse_pantry  # noqa: E402
from pantryrag.retrieval import _head_match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def ok(pred, gold_alts):
    """True if `pred` matches one of the "|"-separated gold answers (up to variety words)."""
    if pred is None:
        return False
    golds = [canonical_form(g) for g in gold_alts.split("|")]
    return any(pred == g or _head_match(pred, g) for g in golds)


def main():
    """Run the cumulative normalisation ablation and the pantry-parsing evaluation."""
    RES.mkdir(exist_ok=True)
    recipes, vocab = corpus.load()
    names = [k for k, c in vocab.items() if c >= 2]
    rows = list(csv.DictReader(open(ROOT / "data/eval/normalization_gold.csv")))

    # each config switches on one more stage than the previous (cumulative ablation)
    configs = {
        "exact match only": dict(use_clean=False, use_singular=False, use_synonyms=False, use_fuzzy=False),
        "+ cleaning": dict(use_clean=True, use_singular=False, use_synonyms=False, use_fuzzy=False),
        "+ singularisation": dict(use_clean=True, use_singular=True, use_synonyms=False, use_fuzzy=False),
        "+ synonym map": dict(use_clean=True, use_singular=True, use_synonyms=True, use_fuzzy=False),
        "+ fuzzy (full)": dict(use_clean=True, use_singular=True, use_synonyms=True, use_fuzzy=True),
        "full w/o synonyms": dict(use_clean=True, use_singular=True, use_synonyms=False, use_fuzzy=True),
    }
    out = {"n": len(rows), "corpus_source": (corpus.PROCESSED / "SOURCE").read_text().strip(),
           "ablation": {}}
    errors = []
    for name, cfg in configs.items():
        n = Normalizer(names, vocab, **cfg)
        res = {"all": [0, 0], "recipe_phrase": [0, 0], "user_term": [0, 0]}
        for r in rows:
            p = n.normalize(r["surface"]).canonical
            c = ok(p, r["gold"])
            for key in ("all", r["kind"]):
                res[key][0] += c
                res[key][1] += 1
            if name == "+ fuzzy (full)" and not c:
                errors.append({"surface": r["surface"], "gold": r["gold"], "pred": p,
                               "method": n.normalize(r["surface"]).method})
        out["ablation"][name] = {k: round(a / b, 3) for k, (a, b) in res.items()}
        print(f"{name:22s} " + "  ".join(f"{k}={a/b:.3f}" for k, (a, b) in res.items()))

    # pantry query parsing
    norm = Normalizer(names, vocab)
    qs = [json.loads(l) for l in open(ROOT / "data/eval/pantry_queries_gold.jsonl")]
    tp = fp = fn = 0
    cons_ok = cons_n = 0
    for g in qs:
        q = parse_pantry(g["text"], norm)
        gold = list(g["ingredients"])
        # one-to-one matching: each gold ingredient can be credited to only one
        # prediction, so duplicates count as false positives
        matched = set()
        for p in q.ingredients:
            hit = next((i for i, alts in enumerate(gold) if i not in matched and ok(p, alts)), None)
            if hit is None:
                fp += 1
            else:
                matched.add(hit)
                tp += 1
        fn += len(gold) - len(matched)
        for key in ("diet", "max_minutes", "cuisine"):
            cons_n += 1
            cons_ok += getattr(q, key) == g.get(key)
        cons_n += 1
        gex = g.get("exclude", [])
        cons_ok += len(q.exclude) == len(gex) and all(any(ok(e, a) for e in q.exclude) for a in gex)
    P, R = tp / (tp + fp), tp / (tp + fn)
    out["pantry_parsing"] = {"n_queries": len(qs), "ingredient_precision": round(P, 3),
                             "ingredient_recall": round(R, 3), "ingredient_f1": round(2 * P * R / (P + R), 3),
                             "constraint_field_accuracy": round(cons_ok / cons_n, 3)}
    print("pantry parsing:", out["pantry_parsing"])
    (RES / "normalization.json").write_text(json.dumps(out, indent=2))
    with open(RES / "normalization_errors.csv", "w") as f:
        w = csv.DictWriter(f, ["surface", "gold", "pred", "method"])
        w.writeheader()
        w.writerows(errors)


if __name__ == "__main__":
    main()
