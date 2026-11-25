"""Stand-in corpus: convert the group's scraped BBC Good Food CSV (10k recipes)
into the same raw JSONL schema the RecipeDB fetcher writes.

Used only while the RecipeDB API key has no quota left. Ingredient names come
from the NER column of q1b_ingredients_final.csv.

    python scripts/import_bbc.py --dir /path/to/Group44_A1_CGAS/Q1
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "bbc_recipes.jsonl"


def iso_minutes(s) -> float | None:
    if not isinstance(s, str):
        return None
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", s.strip())
    if not m or not (m.group(1) or m.group(2)):
        return None
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    raw = pd.read_csv(d / "recipes_raw.csv")
    ner = pd.read_csv(d / "q1b_ingredients_final.csv")
    names = defaultdict(list)
    phrases = defaultdict(list)
    for rid, ph, nm in ner.itertuples(index=False):
        if isinstance(nm, str) and nm.strip():
            names[rid].append(nm.strip().lower())
            phrases[rid].append(ph)
    n = 0
    with open(OUT, "w") as f:
        for i, row in enumerate(raw.itertuples(index=False), start=1):
            rid = f"Recipe {i}"
            try:
                steps_raw = ast.literal_eval(row.instructions)
            except (ValueError, SyntaxError):
                continue
            steps = []
            for para in steps_raw:
                steps += [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if len(s.strip()) > 3]
            rec = {
                "Recipe_id": f"bbc{i}", "Recipe_title": row.recipe_name, "url": row.recipe_url,
                "Region": row.cuisine if isinstance(row.cuisine, str) else "",
                "Sub_region": "", "Continent": "", "Source": "BBC Good Food (stand-in)",
                "img_url": None, "servings": None if pd.isna(row.servings) else str(row.servings),
                "CaloriesNum": None, "TotalTimeNum": iso_minutes(row.prep_time),
                "ingredients": [{"name": x} for x in names.get(rid, [])],
                "ingredient_phrases": phrases.get(rid, []),
                "steps": steps,
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n} recipes to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
