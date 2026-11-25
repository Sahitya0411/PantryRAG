"""D2 evaluation: BM25 vs dense vs hybrid (RRF) vs hybrid+coverage rerank.

Synthetic pantry queries: sample a recipe, hide it (query has no title), and
use 3-5 of its non-staple ingredients as the pantry. Metrics: Recall@k and MRR
for retrieving the hidden recipe, plus user-facing quality of the top-5
(mean pantry coverage, mean #missing ingredients).

Reranker weights are tuned on a dev split and reported on a disjoint test split.
Also measures hard-constraint violations with vs. without metadata filters.

    python scripts/eval_retrieval.py --n-test 500 --n-dev 200
"""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pantryrag import corpus  # noqa: E402
from pantryrag.lexicon import STAPLES  # noqa: E402
from pantryrag.pantry_parser import PantryQuery  # noqa: E402
from pantryrag.retrieval import Retriever, _head_match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODES = ["bm25", "dense", "hybrid", "hybrid_rerank"]
KS = [1, 5, 10, 20]


def make_queries(recipes, n, rng, exclude=()):
    pool = [r for r in recipes if r.id not in exclude and
            len([i for i in r.ingredients if i not in STAPLES and
                 not any(_head_match(i, s) for s in STAPLES)]) >= 5]
    out = []
    for r in rng.sample(pool, n):
        need = [i for i in r.ingredients if i not in STAPLES and not any(_head_match(i, s) for s in STAPLES)]
        k = rng.randint(3, 5)
        out.append((r.id, PantryQuery(text="", ingredients=rng.sample(need, k))))
    return out


def evaluate(R, queries, mode, depth=100, **kw):
    ranks, cov5, miss5 = [], [], []
    for rid, q in queries:
        hits = R.search(q, k=depth, mode=mode, use_filters=False, depth=depth, **kw)
        ids = [h.recipe.id for h in hits]
        ranks.append(ids.index(rid) + 1 if rid in ids else None)
        top = hits[:5]
        cov5.append(np.mean([h.coverage for h in top]) if top else 0)
        miss5.append(np.mean([len(h.missing) for h in top]) if top else 0)
    res = {f"R@{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in KS}
    res["MRR"] = float(np.mean([1 / r if r else 0 for r in ranks]))
    res["top5_coverage"] = float(np.mean(cov5))
    res["top5_missing"] = float(np.mean(miss5))
    return res


def constraint_eval(R, recipes, rng, n=200):
    """Queries with a diet/time constraint taken from a sampled recipe; how often
    do unfiltered top-10 results violate the constraint?"""
    out = {}
    for use in (False, True):
        viol, tot = 0, 0
        r2 = random.Random(1)
        for r in r2.sample(recipes, n):
            need = [i for i in r.ingredients if i not in STAPLES][:4]
            diet = r2.choice(["vegetarian", "vegan", None])
            q = PantryQuery("", ingredients=need, diet=diet,
                            max_minutes=r2.choice([20, 30, 45, None]))
            for h in R.search(q, k=10, use_filters=use):
                tot += 1
                viol += not R.passes(h.recipe, q)
        out["with_filters" if use else "no_filters"] = round(viol / max(tot, 1), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--n-dev", type=int, default=200)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    recipes, _ = corpus.load()
    R = Retriever.load(recipes)

    test = make_queries(recipes, args.n_test, rng)
    dev = make_queries(recipes, args.n_dev, rng, exclude={rid for rid, _ in test})

    # tune reranker on dev (objective: MRR + top-5 coverage)
    grid = []
    for w_cov, w_use, mp in itertools.product([0.2, 0.3, 0.4], [0.2, 0.3, 0.4], [0.0, 0.02, 0.05]):
        if w_cov + w_use >= 0.9:
            continue
        r = evaluate(R, dev, "hybrid_rerank", w_cov=w_cov, w_use=w_use, miss_pen=mp)
        grid.append(({"w_cov": w_cov, "w_use": w_use, "miss_pen": mp}, r))
        print("dev", grid[-1][0], {k: round(v, 3) for k, v in r.items()}, flush=True)
    best = max(grid, key=lambda g: g[1]["MRR"] + 0.5 * g[1]["top5_coverage"])[0]
    print("best rerank params:", best)

    results = {"corpus_source": (corpus.PROCESSED / "SOURCE").read_text().strip(),
               "n_recipes": len(recipes), "n_test": len(test), "n_dev": len(dev),
               "rerank_params": best, "test": {}}
    for mode in MODES:
        kw = best if mode == "hybrid_rerank" else {}
        results["test"][mode] = evaluate(R, test, mode, **kw)
        print(mode, {k: round(v, 3) for k, v in results["test"][mode].items()}, flush=True)
    results["constraint_violation_rate_top10"] = constraint_eval(R, recipes, rng)
    print("constraint violations:", results["constraint_violation_rate_top10"])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/retrieval.json").write_text(json.dumps(results, indent=2))
    (ROOT / "results/rerank_params.json").write_text(json.dumps(best))


if __name__ == "__main__":
    main()
