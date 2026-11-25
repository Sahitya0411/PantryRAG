"""D1 step 1: download a RecipeDB corpus (metadata + ingredients + steps).

The API caps pages at 10 recipes and rate-limits aggressively (HTTP 429), so
this script throttles every request (--interval seconds) and is fully
resumable: metadata pages and recipe details are appended to JSONL files and
skipped on re-run.

    python scripts/fetch_corpus.py --random-pages 200 --indian-pages 60 --interval 1.0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pantryrag.recipedb import RecipeDBClient, RecipeDBError  # noqa: E402

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
META_KEYS = ["Recipe_id", "Recipe_title", "Region", "Sub_region", "Continent", "Source",
             "img_url", "servings", "CaloriesNum", "TotalTimeNum", "PrepTimeNum",
             "CookTimeNum", "Processes", "UtensilsList", "Protein (g)",
             "Total lipid (fat) (g)", "Carbohydrate, by difference (g)", "Energy (kcal)"]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def collect_metadata(client: RecipeDBClient, random_pages: int, indian_pages: int,
                     seed: int) -> list[dict]:
    pages_path = RAW / "metadata_pages.jsonl"
    done_pages = {(r["kind"], r["page"]) for r in _read_jsonl(pages_path)}
    total_pages = 11809  # 118,083 recipes / 10 (from API meta)
    rng = random.Random(seed)
    jobs = [("all", p) for p in rng.sample(range(1, total_pages + 1), random_pages)]
    jobs += [("Indian Subcontinent", p) for p in range(1, indian_pages + 1)]
    todo = [j for j in jobs if j not in done_pages]
    print(f"metadata: {len(done_pages)} pages cached, {len(todo)} to fetch", flush=True)
    with open(pages_path, "a") as f:
        for i, (kind, p) in enumerate(todo):
            try:
                rows = (client.list_recipes(p, 10)[0] if kind == "all"
                        else client.list_by_region(kind, p, 10))
            except RecipeDBError as e:
                print("  page failed", kind, p, e, file=sys.stderr, flush=True)
                continue
            f.write(json.dumps({"kind": kind, "page": p,
                                "rows": [{k: r.get(k) for k in META_KEYS} for r in rows]}) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  metadata page {i}/{len(todo)}", flush=True)
    metas: dict[str, dict] = {}
    for rec in _read_jsonl(pages_path):
        for r in rec["rows"]:
            metas.setdefault(str(r["Recipe_id"]), r)
    return list(metas.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-pages", type=int, default=200)
    ap.add_argument("--indian-pages", type=int, default=60)
    ap.add_argument("--interval", type=float, default=2.6, help="seconds between API calls")
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    client = RecipeDBClient(min_interval=args.interval)
    metas = collect_metadata(client, args.random_pages, args.indian_pages, args.seed)
    print(f"{len(metas)} recipes in metadata", flush=True)

    out_path = RAW / "recipes.jsonl"
    done = {r["Recipe_id"] for r in _read_jsonl(out_path)}
    todo = [m for m in metas if str(m["Recipe_id"]) not in done]
    print(f"details: {len(done)} cached, {len(todo)} to fetch", flush=True)
    n_fail = 0
    with open(out_path, "a") as f:
        for i, m in enumerate(todo):
            rid = str(m["Recipe_id"])
            try:
                full = client.full_recipe(rid)  # 1 call: metadata + ingredients + steps
            except RecipeDBError as e:
                n_fail += 1
                print("  recipe failed", rid, e, file=sys.stderr, flush=True)
                continue
            if not full:
                n_fail += 1
                continue
            extra = {k: full["meta"].get(k) for k in META_KEYS if full["meta"].get(k) is not None}
            f.write(json.dumps({**m, **extra, "Recipe_id": rid, "ingredients": full["ingredients"],
                                "ingredient_phrases": [i["phrase"] for i in full["ingredients"]],
                                "steps": full["steps"]}) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"  details {i}/{len(todo)} (failures {n_fail})", flush=True)
    print(f"done; failures={n_fail}", flush=True)


if __name__ == "__main__":
    main()
