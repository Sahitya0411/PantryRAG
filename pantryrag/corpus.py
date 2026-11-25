"""D1: clean raw RecipeDB records into the document store used by retrieval."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .diet import recipe_diets
from .lexicon import SYNONYMS
from .normalize import canonical_form

# fold regional names inside the corpus too (e.g. "coriander" leaves -> cilantro is NOT
# applied: in recipes "coriander" is usually the spice). Only unambiguous folds here.
CORPUS_SYNONYMS = {canonical_form(k): canonical_form(v) for k, v in SYNONYMS.items()
                   if k not in {"coriander", "pepper", "oil", "sugar", "chili", "chilli",
                                "dal", "daal", "mirch", "basmati", "pasta", "noodles",
                                "soy", "salt and pepper"}}

ROOT = Path(__file__).resolve().parent.parent
RAW_SOURCES = {
    "recipedb": ROOT / "data" / "raw" / "recipes.jsonl",     # scripts/fetch_corpus.py
    "bbc": ROOT / "data" / "raw" / "bbc_recipes.jsonl",      # scripts/import_bbc.py (stand-in)
}
PROCESSED = ROOT / "data" / "processed"


@dataclass
class Recipe:
    """One cleaned recipe: canonical ingredient names, steps, and the metadata
    (time, calories, cuisine, diet labels) used by the retrieval filters."""
    id: str
    title: str
    ingredients: list[str]
    steps: list[str]
    region: str = ""
    sub_region: str = ""
    continent: str = ""
    total_minutes: float | None = None
    calories: float | None = None  # per serving
    servings: str | None = None
    source: str = ""
    img_url: str | None = None
    diets: dict = field(default_factory=dict)
    url: str | None = None
    tags: list = field(default_factory=list)          # site keywords (scraped)
    source_diets: list = field(default_factory=list)  # site's own diet tags (scraped)

    @property
    def cuisine_text(self) -> str:
        """Region, sub-region and continent as one lowercase string (for the cuisine filter)."""
        return " | ".join(x for x in (self.region, self.sub_region, self.continent) if x).lower()

    def as_document(self) -> str:
        """Text used for dense embedding: the ingredient list only. Adding the
        title and cuisine halved dense MRR on pantry queries (0.50 -> 0.26), since
        queries are ingredient lists too (see report, dense document ablation)."""
        return ", ".join(self.ingredients)

    def as_context(self, max_steps: int = 12) -> str:
        """Text shown to the LLM as a retrieved document."""
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.steps[:max_steps]))
        t = f"{int(self.total_minutes)} min" if self.total_minutes else "unknown time"
        return (f"[R{self.id}] {self.title} ({self.sub_region or self.region}; {t})\n"
                f"Ingredients: {', '.join(self.ingredients)}\nSteps:\n{steps}")


def _num(x):
    """Parse a positive number; RecipeDB uses 0 / blanks for "unknown", so those become None."""
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


SCRAPED = ROOT / "data" / "raw" / "bbc_scraped.jsonl"


def _iso_minutes(s):
    """Convert an ISO-8601 duration such as "PT1H30M" (schema.org format) to minutes."""
    # optional hours group then optional minutes group; reject strings with neither
    m = re.fullmatch(r"P(?:T)?(?:(\d+)H)?(?:(\d+)M)?", str(s or "").strip())
    if not m or not (m.group(1) or m.group(2)):
        return None
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)


def _load_scraped() -> dict[str, dict]:
    """Page metadata scraped from BBC Good Food (scripts/scrape_bbc.py), keyed by URL."""
    if not SCRAPED.exists():
        return {}
    out = {}
    for line in SCRAPED.read_text().splitlines():
        r = json.loads(line)
        # calories arrive as text like "412 calories"; keep only the leading number
        cal = re.match(r"(\d+)", str((r.get("nutrition") or {}).get("calories") or ""))
        diets = r.get("suitableForDiet") or []
        diets = [diets] if isinstance(diets, str) else diets
        # diet tags are schema.org URLs ("https://schema.org/VegetarianDiet") -> "vegetarian"
        out[r["url"]] = {
            "total_minutes": _iso_minutes(r.get("totalTime")),
            "calories": float(cal.group(1)) if cal else None,
            "tags": [t.strip() for t in str(r.get("keywords") or "").split(",") if t.strip()],
            "source_diets": [d.rsplit("/", 1)[-1].replace("Diet", "").lower() for d in diets],
            "cuisine": r.get("recipeCuisine"),
        }
    return out


def build(source: str = "recipedb", min_ingredients: int = 3) -> tuple[list[Recipe], Counter]:
    """Clean the raw JSONL for `source` into Recipe objects.

    Ingredient names are canonicalised, recipes with fewer than `min_ingredients`
    ingredients, no steps or no title are dropped, and duplicates are removed.
    Returns (recipes, ingredient frequency counter)."""
    recipes, freq = [], Counter()
    scraped = _load_scraped() if source == "bbc" else {}
    seen_titles = set()
    for line in RAW_SOURCES[source].read_text().splitlines():
        r = json.loads(line)
        ings = []
        for i in r.get("ingredients", []):
            # canonical name, then fold safe synonyms so one ingredient = one token
            c = canonical_form(i["name"])
            c = CORPUS_SYNONYMS.get(c, c)
            if c and c not in ings:
                ings.append(c)
        steps = [s for s in r.get("steps", []) if len(s) > 3]
        title = (r.get("Recipe_title") or "").strip()
        if len(ings) < min_ingredients or not steps or not title:
            continue
        # drop duplicate recipes: same title and same ingredient set
        key = (title.lower(), tuple(sorted(ings)))
        if key in seen_titles:
            continue
        seen_titles.add(key)
        rec = Recipe(
            id=str(r["Recipe_id"]), title=title, ingredients=ings, steps=steps,
            region=r.get("Region") or "", sub_region=r.get("Sub_region") or "",
            continent=r.get("Continent") or "", total_minutes=_num(r.get("TotalTimeNum")),
            calories=_num(r.get("CaloriesNum")), servings=r.get("servings"),
            source=r.get("Source") or "", img_url=r.get("img_url"),
            diets=recipe_diets(ings), url=r.get("url"))
        extra = scraped.get(r.get("url") or "")
        if extra:  # real total time / calories / tags from the live page
            rec.total_minutes = extra["total_minutes"] or rec.total_minutes
            rec.calories = extra["calories"] or rec.calories
            rec.tags = extra["tags"]
            rec.source_diets = extra["source_diets"]
            if extra["cuisine"] and not rec.region:
                rec.region = extra["cuisine"]
        recipes.append(rec)
        freq.update(ings)
    return recipes, freq


def save(recipes: list[Recipe], freq: Counter) -> None:
    """Write the cleaned recipes and the ingredient vocabulary (with counts) to data/processed/."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "recipes.json").write_text(json.dumps([asdict(r) for r in recipes]))
    (PROCESSED / "vocab.json").write_text(json.dumps(dict(freq.most_common())))


def load() -> tuple[list[Recipe], dict[str, int]]:
    """Read the processed corpus back. Returns (recipes, {ingredient: frequency})."""
    rows = json.loads((PROCESSED / "recipes.json").read_text())
    vocab = json.loads((PROCESSED / "vocab.json").read_text())
    return [Recipe(**r) for r in rows], vocab
