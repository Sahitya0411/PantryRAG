"""D3: substitution suggestions for missing ingredients.

Two sources, in priority order:
  1. A curated table of well-known culinary substitutions (with ratios/notes).
  2. A data-driven fallback: ingredients that occur in *similar recipe contexts*
     but rarely *together* (substitutes, not complements). Each ingredient is a
     PPMI co-occurrence vector over the corpus; candidates are ranked by cosine
     similarity of those vectors, penalised by direct co-occurrence.

Suggestions from the user's own pantry are preferred ("you have X, use it
instead of Y"); others are flagged as needing a purchase.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lexicon import STAPLES

CURATED: dict[str, list[tuple[str, str]]] = {
    "heavy cream": [("milk", "use whole milk + 1 tbsp butter per cup"), ("greek yogurt", "stir in off the heat"),
                    ("coconut cream", "dairy-free; adds coconut flavour"), ("cashew", "blend soaked cashews with water")],
    "cream": [("milk", "milk + a little butter"), ("yogurt", "add off the heat to avoid splitting"),
              ("coconut milk", "dairy-free")],
    "sour cream": [("greek yogurt", "1:1"), ("yogurt", "1:1, slightly thinner")],
    "yogurt": [("sour cream", "1:1"), ("buttermilk", "thinner"), ("lemon", "milk + lemon juice for marinades")],
    "buttermilk": [("milk", "1 cup milk + 1 tbsp lemon juice/vinegar, rest 5 min"), ("yogurt", "thin with water")],
    "butter": [("ghee", "1:1"), ("vegetable oil", "use 3/4 the amount"), ("olive oil", "3/4 the amount, savoury dishes")],
    "ghee": [("butter", "1:1, lower smoke point"), ("vegetable oil", "1:1")],
    "milk": [("soy milk", "1:1"), ("almond milk", "1:1"), ("coconut milk", "richer")],
    "paneer": [("tofu", "firm tofu, pressed; vegan"), ("halloumi", "grills well, saltier"), ("ricotta", "softer")],
    "tofu": [("paneer", "not vegan"), ("chickpea", "different texture")],
    "egg": [("flaxseed", "1 tbsp ground flax + 3 tbsp water per egg (baking)"), ("banana", "1/4 cup mashed per egg (sweet bakes)"),
            ("yogurt", "1/4 cup per egg (baking)"), ("chickpea flour", "3 tbsp + 3 tbsp water (savoury)")],
    "all purpose flour": [("whole wheat flour", "denser; add a little extra liquid"), ("self raising flour", "omit baking powder")],
    "self raising flour": [("all purpose flour", "add 1.5 tsp baking powder per cup")],
    "baking powder": [("baking soda", "1/4 tsp soda + 1/2 tsp cream of tartar per tsp")],
    "white sugar": [("honey", "use 3/4 and reduce liquid"), ("brown sugar", "1:1"), ("jaggery", "1:1, caramel notes")],
    "brown sugar": [("white sugar", "add 1 tbsp molasses per cup"), ("jaggery", "1:1")],
    "honey": [("maple syrup", "1:1, vegan"), ("white sugar", "use 1.25x and add liquid")],
    "lemon": [("lime", "1:1"), ("vinegar", "for acidity; use half")],
    "lemon juice": [("lime juice", "1:1"), ("vinegar", "half the amount")],
    "lime": [("lemon", "1:1")],
    "vinegar": [("lemon juice", "1:1")],
    "white wine": [("vegetable stock", "+ a splash of vinegar"), ("chicken stock", "+ a splash of lemon")],
    "red wine": [("vegetable stock", "+ a splash of balsamic vinegar")],
    "wine": [("vegetable stock", "+ a splash of vinegar")],
    "chicken stock": [("vegetable stock", "1:1, vegetarian"), ("water", "+ bouillon or extra seasoning")],
    "beef stock": [("vegetable stock", "1:1"), ("chicken stock", "1:1")],
    "vegetable stock": [("water", "+ salt and herbs"), ("chicken stock", "not vegetarian")],
    "shallot": [("onion", "use a little less"), ("red onion", "milder raw")],
    "onion": [("shallot", "1:1"), ("leek", "milder"), ("green onion", "add late")],
    "red onion": [("onion", "1:1")],
    "green onion": [("onion", "use less; cook briefly"), ("chive", "garnish")],
    "garlic": [("garlic powder", "1/8 tsp per clove"), ("shallot", "milder")],
    "ginger": [("ground ginger", "1/4 tsp per tbsp fresh")],
    "cilantro": [("parsley", "milder"), ("mint", "different but fresh")],
    "coriander": [("cumin", "earthier"), ("garam masala", "warmer")],
    "parsley": [("cilantro", "stronger"), ("basil", "sweeter")],
    "basil": [("parsley", "milder"), ("spinach", "for pesto bulk")],
    "spinach": [("kale", "cook a little longer"), ("swiss chard", "1:1"), ("fenugreek leaf", "bitter; use less")],
    "kale": [("spinach", "cook briefly")],
    "tomato": [("canned tomato", "1:1"), ("tomato paste", "1 tbsp paste + water per tomato")],
    "canned tomato": [("tomato", "fresh, chopped; cook longer"), ("tomato paste", "dilute with water")],
    "tomato paste": [("tomato", "cook down 2-3 fresh tomatoes"), ("ketchup", "sweeter")],
    "rice": [("quinoa", "1:1"), ("couscous", "quicker"), ("cauliflower", "riced, low-carb")],
    "basmati rice": [("rice", "any long-grain rice")],
    "pasta": [("noodle", "1:1"), ("rice", "different dish style")],
    "spaghetti": [("pasta", "any shape"), ("noodle", "1:1")],
    "noodle": [("spaghetti", "1:1"), ("rice", "serve as rice bowl")],
    "potato": [("sweet potato", "sweeter"), ("cauliflower", "lower-carb")],
    "sweet potato": [("potato", "less sweet"), ("butternut squash", "1:1")],
    "chicken": [("paneer", "vegetarian"), ("tofu", "vegan"), ("chickpea", "vegan, add late"), ("turkey", "1:1")],
    "chicken breast": [("chicken thigh", "juicier"), ("tofu", "vegan"), ("paneer", "vegetarian")],
    "beef": [("lamb", "1:1"), ("mushroom", "vegetarian, meaty texture"), ("lentil", "vegan mince")],
    "ground beef": [("lentil", "vegan"), ("ground lamb", "1:1"), ("mushroom", "finely chopped")],
    "lamb": [("beef", "1:1"), ("chickpea", "vegetarian")],
    "bacon": [("smoked paprika", "for smokiness"), ("mushroom", "fried crisp")],
    "shrimp": [("chicken", "1:1"), ("tofu", "vegan")],
    "fish": [("tofu", "vegan"), ("paneer", "vegetarian")],
    "parmesan": [("cheddar", "milder"), ("nutritional yeast", "vegan")],
    "cheddar": [("parmesan", "stronger"), ("mozzarella", "milder")],
    "mozzarella": [("paneer", "doesn't melt the same"), ("cheddar", "stronger")],
    "soy sauce": [("tamari", "gluten-free"), ("salt", "+ a little vinegar")],
    "fish sauce": [("soy sauce", "vegetarian")],
    "bell pepper": [("zucchini", "different flavour"), ("carrot", "for crunch/colour")],
    "pepper": [("zucchini", "different flavour"), ("carrot", "crunch/colour")],
    "zucchini": [("eggplant", "1:1"), ("bell pepper", "1:1")],
    "eggplant": [("zucchini", "1:1"), ("mushroom", "meaty")],
    "chickpea": [("kidney bean", "1:1"), ("lentil", "1:1, softer")],
    "kidney bean": [("chickpea", "1:1"), ("black bean", "1:1")],
    "lentil": [("red lentil", "1:1, cooks faster"), ("chickpea", "1:1")],
    "red lentil": [("lentil", "cooks slower"), ("mung bean", "1:1")],
    "coconut milk": [("heavy cream", "not vegan"), ("yogurt", "tangier; add off heat"), ("milk", "+ desiccated coconut")],
    "garam masala": [("cumin", "+ coriander + cinnamon pinch"), ("curry powder", "1:1")],
    "curry powder": [("garam masala", "+ turmeric")],
    "cumin": [("coriander", "milder"), ("garam masala", "warmer")],
    "chili powder": [("paprika", "milder"), ("red chile powder", "hotter")],
    "red chile powder": [("paprika", "milder"), ("chili powder", "1:1")],
    "green chile pepper": [("red chile powder", "use a pinch"), ("chilli", "1:1")],
    "chilli": [("red chile powder", "pinch"), ("chili flake", "pinch")],
    "mushroom": [("eggplant", "meaty texture"), ("zucchini", "milder")],
    "breadcrumb": [("oat", "blitzed"), ("cracker", "crushed")],
    "bread crumb": [("oat", "blitzed"), ("cracker", "crushed")],
    "cornstarch": [("all purpose flour", "2x amount"), ("arrowroot", "1:1")],
    "maple syrup": [("honey", "1:1"), ("brown sugar", "dissolved")],
    "almond": [("cashew", "1:1"), ("peanut", "1:1")],
    "cashew": [("almond", "1:1"), ("peanut", "1:1")],
    "peanut butter": [("almond butter", "1:1"), ("tahini", "1:1, nut-free")],
    "tahini": [("peanut butter", "1:1")],
}


@dataclass
class Substitution:
    """One suggested replacement for a missing ingredient, with a usage note and
    whether the user already has it."""
    missing: str
    substitute: str
    note: str
    in_pantry: bool
    source: str  # curated | corpus


class Substituter:
    """Suggests substitutes from the curated table, backed up by corpus statistics.

    Only ingredients seen in at least `min_count` recipes get corpus vectors (rare
    ones give noisy statistics). `encoder` is a sentence-embedding model; pass the
    retriever's to avoid loading it twice."""

    def __init__(self, recipes, min_count: int = 15, encoder=None):
        from collections import Counter
        cnt = Counter(i for r in recipes for i in r.ingredients)
        self.vocab = [i for i, c in cnt.items() if c >= min_count and i not in STAPLES]
        self.idx = {v: k for k, v in enumerate(self.vocab)}
        n = len(self.vocab)
        # C[a, b] = number of recipes that contain both ingredient a and ingredient b
        C = np.zeros((n, n), np.float32)
        for r in recipes:
            ids = [self.idx[i] for i in r.ingredients if i in self.idx]
            for a in ids:
                C[a, ids] += 1
        np.fill_diagonal(C, 0)
        self.cooc = C
        # PPMI = max(0, log P(a,b) / (P(a) P(b))): how much more often two ingredients
        # co-occur than chance. Negative values are clipped (unreliable for sparse
        # counts); the epsilons avoid division by zero and log(0).
        tot = C.sum()
        row = C.sum(1, keepdims=True) + 1e-9
        col = C.sum(0, keepdims=True) + 1e-9
        pmi = np.log((C * tot) / (row * col) + 1e-12)
        ppmi = np.maximum(pmi, 0)
        # unit-length rows, so a dot product between two rows is their cosine similarity
        norms = np.linalg.norm(ppmi, axis=1, keepdims=True) + 1e-9
        self.vec = ppmi / norms
        self.count = np.array([cnt[v] for v in self.vocab], np.float32)
        # semantic similarity of the ingredient *names* (keeps suggestions within
        # the same kind of food: fat->fat, herb->herb, not spice->rice)
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.name_vec = encoder.encode(self.vocab, normalize_embeddings=True,
                                       batch_size=128).astype(np.float32)

    def corpus_candidates(self, ing: str, k: int = 10) -> list[tuple[str, float]]:
        """Top `k` data-driven substitutes for `ing` as (name, score) pairs.

        Score = geometric mean of context similarity and name similarity, minus how
        often the two already appear in the same recipe."""
        if ing not in self.idx:
            return []
        i = self.idx[ing]
        ctx = self.vec @ self.vec[i]                 # second-order context similarity
        name = self.name_vec @ self.name_vec[i]      # same kind of food?
        # substitutes appear in similar contexts but rarely in the same recipe
        together = self.cooc[i] / (np.minimum(self.count, self.count[i]) + 1e-9)
        # geometric mean: a candidate must score on BOTH signals, not just one
        score = np.sqrt(np.clip(ctx, 0, None) * np.clip(name, 0, None)) - together
        score[i] = -np.inf
        for j, v in enumerate(self.vocab):  # drop variants of the same item
            if v in ing or ing in v:
                score[j] = -np.inf
        order = np.argsort(-score)[:k]
        return [(self.vocab[j], float(score[j])) for j in order if np.isfinite(score[j])]

    def suggest(self, missing: str, pantry: list[str], diet: str | None = None,
                k: int = 2) -> list[Substitution]:
        """Up to `k` substitutes for `missing`, respecting `diet` and preferring items
        already in `pantry`, then curated suggestions over corpus-derived ones."""
        from .diet import recipe_diets

        def diet_ok(name: str) -> bool:
            return diet is None or recipe_diets([name]).get(diet, True)

        def have(name: str) -> bool:
            return any(name == p or name in p.split() or p in name.split() for p in pantry)

        out: list[Substitution] = []
        cands = [(s, note, "curated") for s, note in CURATED.get(missing, [])]
        if not cands and len(missing.split()) == 2:  # head noun: "white vinegar" -> "vinegar"
            cands = [(s, note, "curated") for s, note in CURATED.get(missing.split()[-1], [])]
        for s, sc in self.corpus_candidates(missing, k=8):
            # keep only confident corpus suggestions (score threshold 0.4)
            if sc > 0.4:
                cands.append((s, f"plays a similar role in other recipes (score {sc:.2f})", "corpus"))
        seen = set()
        # keep diet-safe candidates, first occurrence only (seen.add() returns None,
        # so it records the name without filtering it out)
        cands = [c for c in cands if diet_ok(c[0]) and not (c[0] in seen or seen.add(c[0]))]
        # pantry items first, then curated before corpus
        cands.sort(key=lambda c: (not have(c[0]), c[2] != "curated"))
        for s, note, src in cands[:k]:
            out.append(Substitution(missing, s, note, have(s), src))
        return out
