"""Rule-based diet labelling of a recipe from its canonical ingredient names.

RecipeDB ships diet flags (vegan, lacto_vegetarian, ...) but in the records we
pulled they are 0.0 even for obviously vegetarian dishes (e.g. Palak Paneer),
so we derive diet classes from ingredients instead.
"""
from __future__ import annotations

import re

from .lexicon import DAIRY, EGG, MEAT, NOT_ANIMAL, OTHER_ANIMAL, SEAFOOD


def _has(name: str, words: set[str]) -> bool:
    # blank out plant phrases that contain an animal word ("coconut milk",
    # "vegetable stock"), then look for animal words in what is left, so that
    # "chicken vegetable stock" is still caught as meat.
    rest = name
    for n in sorted(NOT_ANIMAL, key=len, reverse=True):
        rest = rest.replace(n.rstrip("_"), " ")
    return any(re.search(rf"\b{re.escape(w)}s?\b", rest) for w in words)


def ingredient_classes(name: str) -> set[str]:
    c = set()
    if _has(name, MEAT):
        c.add("meat")
    if _has(name, SEAFOOD):
        c.add("seafood")
    if _has(name, EGG):
        c.add("egg")
    if _has(name, DAIRY):
        c.add("dairy")
    if _has(name, OTHER_ANIMAL):
        c.add("honey")
    return c


def recipe_diets(ingredients: list[str]) -> dict[str, bool]:
    classes = set()
    for i in ingredients:
        classes |= ingredient_classes(i)
    return {
        "vegan": not classes,
        "vegetarian": not ({"meat", "seafood"} & classes),
        "eggless_vegetarian": not ({"meat", "seafood", "egg"} & classes),
        "pescatarian": "meat" not in classes,
        "dairy_free": "dairy" not in classes,
    }


DIET_KEYS = ["vegan", "vegetarian", "eggless_vegetarian", "pescatarian", "dairy_free"]
