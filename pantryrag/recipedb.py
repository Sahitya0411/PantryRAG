"""Thin client for the CoSyLab RecipeDB 2.0 API (IIIT Delhi).

Endpoints used (base: http://cosylab.iiitd.edu.in:6969/recipe2-api):
  GET recipe/recipesinfo?page=&limit=          paged recipe metadata (limit <= 10)
  GET recipe/recipesinfo?recipeId=             metadata for one recipe
  GET recipes_cuisine/cuisine/{region}?page=&limit=   paged recipes for a region
  GET recipe-bytitle/recipeByTitle?title=      title search
  GET instructions/{recipe_id}                 cooking steps
  GET recipe-nutri/nutritioninfo?recipeId=     per-ingredient nutrition rows
                                               (this is where ingredient names live)

Auth: `Authorization: Bearer $RECIPEDB_API_KEY` (key from foodoscope.com), read
from the environment or a local `.env` file.
"""
from __future__ import annotations

import json
import re
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://api.foodoscope.com/recipe2-api"  # IIITD mirror: http://cosylab.iiitd.edu.in:6969/recipe2-api


def _load_dotenv() -> None:
    """Load KEY=value lines from the project's .env into os.environ (existing values win)."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class RecipeDBError(RuntimeError):
    """Raised when a RecipeDB request fails permanently (bad key, not found, retries exhausted)."""
    pass


class RecipeDBClient:
    """Minimal HTTP client for RecipeDB with retries, back-off and an optional
    global rate limit (`min_interval` seconds between requests, thread-safe)."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 timeout: int = 30, retries: int = 8, min_interval: float = 0.0):
        _load_dotenv()
        self.api_key = api_key or os.environ.get("RECIPEDB_API_KEY")
        if not self.api_key:
            raise RecipeDBError("RECIPEDB_API_KEY not set (see .env.example)")
        self.base_url = (base_url or os.environ.get("RECIPEDB_BASE_URL") or DEFAULT_BASE_URL)
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.endswith("recipe2-api"):
            self.base_url += "/recipe2-api"
        self.timeout = timeout
        self.retries = retries
        self.min_interval = min_interval  # seconds between requests (global throttle)
        self._last = 0.0
        self._lock = threading.Lock()

    def _throttle(self) -> None:
        """Sleep so consecutive requests are at least `min_interval` seconds apart."""
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict:
        """GET an endpoint and return the decoded JSON.

        429 (rate limit) waits up to 5 minutes and retries; client errors (4xx) fail
        immediately; network errors are retried with a growing delay."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "PantryRAG/1.0",
        })
        last = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "ignore")
                if e.code == 429:  # rate limited: back off hard
                    last = e
                    time.sleep(min(60 * (attempt + 1), 300))
                    continue
                if e.code in (400, 401, 403, 404):
                    raise RecipeDBError(f"HTTP {e.code} for {endpoint}: {body[:200]}")
                last = e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = e
            time.sleep(1.5 * (attempt + 1))
        raise RecipeDBError(f"request failed for {endpoint}: {last}")

    # --- listing -----------------------------------------------------------
    def list_recipes(self, page: int, limit: int = 10) -> tuple[list[dict], dict]:
        """One page of recipe metadata. Returns (rows, paging meta); the API caps limit at 10."""
        res = self._get("recipe/recipesinfo", {"page": page, "limit": min(limit, 10)})
        return res.get("data", []) or [], res.get("meta", {}) or {}

    def list_by_region(self, region: str, page: int, limit: int = 10) -> list[dict]:
        """One page of recipe metadata for a cuisine region (e.g. "Indian Subcontinent")."""
        res = self._get(f"recipes_cuisine/cuisine/{urllib.parse.quote(region)}",
                        {"page": page, "limit": min(limit, 10)})
        return res.get("data", []) or []

    def search_by_title(self, title: str) -> list[dict]:
        """Recipes whose title matches `title`."""
        return self._get("recipe-bytitle/recipeByTitle", {"title": title}).get("data", []) or []

    # --- per recipe --------------------------------------------------------
    def recipe_info(self, recipe_id: str) -> dict | None:
        """Metadata for one recipe, or None if it does not exist."""
        data = self._get("recipe/recipesinfo", {"recipeId": recipe_id}).get("data", [])
        return data[0] if data else None

    def instructions(self, recipe_id: str) -> list[str]:
        """Cooking steps for one recipe as a list of non-empty strings."""
        res = self._get(f"instructions/{recipe_id}")
        data = res.get("data") or res
        steps = data.get("steps", []) if isinstance(data, dict) else []
        return [s.strip() for s in steps if isinstance(s, str) and s.strip()]

    def full_recipe(self, recipe_id: str) -> dict | None:
        """One call for metadata + ingredients (+ raw phrases) + steps.
        Endpoint: GET search-recipe/{id}."""
        res = self._get(f"search-recipe/{recipe_id}")
        d = res.get("data") or {}
        meta = d.get("recipe")
        if not meta:
            return None
        ings, seen = [], set()
        for i in d.get("ingredients") or []:
            name = (i.get("ingredient") or "").strip().lower()
            if name and name not in seen:
                seen.add(name)
                ings.append({"name": name, "phrase": i.get("ingredient_Phrase"),
                             "state": i.get("state"), "quantity": i.get("quantity")})
        steps = []
        for ins in d.get("instructions") or []:
            raw = ins.get("steps")
            # steps come either as a list or as one string with " . " / " ; "
            # between sentences (the API pads punctuation with spaces)
            parts = raw if isinstance(raw, list) else re.split(r"\s[.;]\s|\s[.;]$", raw or "")
            steps += [x.strip(" .;") for x in parts if x and len(x.strip(" .;")) > 3]
        return {"meta": meta, "ingredients": ings, "steps": steps}

    def ingredients(self, recipe_id: str) -> list[dict]:
        """Ingredient rows from the nutrition endpoint (name + key macros)."""
        rows = self._get("recipe-nutri/nutritioninfo", {"recipeId": recipe_id}).get("data", []) or []
        out, seen = [], set()
        for r in rows:
            name = (r.get("ingredientName") or "").strip().lower()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append({
                "name": name,
                "kcal": _f(r.get("Energy (kcal)")),
                "protein_g": _f(r.get("Protein (g)")),
                "fat_g": _f(r.get("Total lipid (fat) (g)")),
                "carb_g": _f(r.get("Carbohydrate, by difference (g)")),
            })
        return out


def _f(x) -> float | None:
    """Parse a number rounded to 3 decimals, or None if missing/invalid."""
    try:
        return round(float(x), 3)
    except (TypeError, ValueError):
        return None
