"""D1: check our ingredient-based diet labels against the site's own diet tags.

BBC Good Food marks some recipes with schema.org suitableForDiet
(VegetarianDiet, VeganDiet) or with "Vegetarian"/"Vegan" keywords. Those tags are
incomplete (an untagged recipe may still be vegetarian), so we report agreement
on the tagged positives, plus how often our labeller calls a recipe with a
meat/fish keyword vegetarian.

    python scripts/eval_diet.py     (after scripts/scrape_bbc.py)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pantryrag import corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MEATY = {"chicken", "beef", "lamb", "pork", "fish", "prawn", "prawns", "salmon", "bacon",
         "sausage", "turkey", "duck", "seafood", "tuna", "cod", "mince", "chorizo", "ham"}


def main():
    recipes, _ = corpus.build("bbc")
    scraped = [r for r in recipes if r.tags or r.source_diets]
    out = {"n_scraped": len(scraped)}
    for diet in ("vegetarian", "vegan"):
        pos = [r for r in scraped if diet in r.source_diets
               or any(t.lower() == diet for t in r.tags)]
        agree = sum(r.diets[diet] for r in pos)
        out[f"{diet}_tagged"] = len(pos)
        out[f"{diet}_agreement"] = round(agree / len(pos), 3) if pos else None
        out[f"{diet}_misses"] = [(r.title, [i for i in r.ingredients][:12]) for r in pos
                                 if not r.diets[diet]][:10]
    meaty = [r for r in scraped if any(t.lower() in MEATY for t in r.tags)]
    false_veg = [r for r in meaty if r.diets["vegetarian"]]
    out["meat_tagged"] = len(meaty)
    out["meat_called_vegetarian"] = round(len(false_veg) / len(meaty), 3) if meaty else None
    out["meat_called_vegetarian_examples"] = [(r.title, r.ingredients[:12]) for r in false_veg[:10]]
    print(json.dumps({k: v for k, v in out.items() if "examples" not in k and "misses" not in k}, indent=2))
    (ROOT / "results/diet_labels.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
