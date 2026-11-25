"""D2: BM25 + dense retrieval, Reciprocal Rank Fusion, hard filters, coverage reranking."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import PROCESSED, Recipe
from .lexicon import STAPLES
from .pantry_parser import CUISINES, PantryQuery

INDEX_DIR = PROCESSED / "index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# modifiers that only name a variety of the same ingredient ("basmati rice" ~ "rice")
VARIETY = {"basmati", "jasmine", "long", "grain", "white", "brown", "red", "green", "yellow",
           "plain", "fresh", "whole", "baby", "roma", "plum", "cherry", "grape", "russet",
           "yukon", "gold", "large", "small", "medium", "boneless", "skinless", "lean",
           "extra", "virgin", "light", "dark", "unsalted", "salted", "low", "fat", "skim",
           "full", "button", "white", "purple", "vine", "ripe", "english", "persian", "hothouse",
           "yellow", "vidalia", "spanish", "sweet_", "italian", "flat", "leaf", "curly",
           "breast", "thigh", "chunk", "regular", "raw", "frozen", "canned", "dried", "firm",
           "extra-firm", "silken", "greek", "sharp", "mild", "aged", "low-fat", "nonfat"}


def _head_match(a: str, b: str) -> bool:
    """True if `a` and `b` are the same ingredient up to variety modifiers."""
    if a == b:
        return True
    wa, wb = set(a.split()), set(b.split())
    # they must share a real (non-variety) word, and every word that differs must be
    # a variety modifier: "basmati rice" ~ "rice", but "sweet potato" !~ "potato"
    core = (wa & wb) - VARIETY
    return bool(core) and (wa ^ wb) <= VARIETY


def covered(recipe_ing: str, pantry: list[str]) -> bool:
    """True if the user can supply `recipe_ing`: it is a staple or matches a pantry item."""
    return (recipe_ing in STAPLES or any(_head_match(recipe_ing, s) for s in STAPLES)
            or any(_head_match(recipe_ing, p) for p in pantry))


def coverage_stats(recipe: Recipe, pantry: list[str]) -> tuple[list[str], list[str], float]:
    """(matched pantry-backed ingredients, missing ingredients, coverage in [0,1]).

    coverage = share of the recipe's non-staple ingredients the user already has."""
    need = [i for i in recipe.ingredients
            if i not in STAPLES and not any(_head_match(i, s) for s in STAPLES)]
    matched = [i for i in need if covered(i, pantry)]
    missing = [i for i in need if i not in matched]
    cov = len(matched) / len(need) if need else 1.0
    return matched, missing, cov


def pantry_use(recipe: Recipe, pantry: list[str]) -> float:
    """Share of the user's pantry items that the recipe actually uses."""
    if not pantry:
        return 0.0
    return sum(any(_head_match(p, i) for i in recipe.ingredients) for p in pantry) / len(pantry)


def bm25_tokens(ings: list[str]) -> list[str]:
    """BM25 tokens for an ingredient list: each multi-word name as one token
    ("bell_pepper") plus its head noun ("pepper"), so exact phrases score highest
    while partial matches still count."""
    toks = []
    for i in ings:
        toks.append(i.replace(" ", "_"))
        if " " in i:
            toks.append(i.split()[-1])  # head noun
    return toks


@dataclass
class Hit:
    """One retrieved recipe: final score, its rank in each retriever list, and how
    well it fits the pantry (matched / missing ingredients, coverage)."""
    recipe: Recipe
    score: float
    ranks: dict = field(default_factory=dict)
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    coverage: float = 0.0


class Retriever:
    """Hybrid recipe retriever: BM25 over ingredient tokens + dense embeddings,
    fused with RRF, restricted by hard filters and optionally reranked by pantry fit."""

    def __init__(self, recipes: list[Recipe], embeddings: np.ndarray | None = None,
                 encoder=None):
        self.recipes = recipes
        self.id2idx = {r.id: i for i, r in enumerate(recipes)}
        self.bm25 = BM25Okapi([bm25_tokens(r.ingredients) for r in recipes])
        self._encoder = encoder
        # Exact inner-product search over L2-normalised embeddings (equivalent to a
        # FAISS IndexFlatIP). We use numpy directly: at ~10^4 docs it is instant, and
        # importing faiss alongside torch segfaults on macOS (duplicate OpenMP).
        self.emb = embeddings if embeddings is not None else self._embed_corpus()

    # ---------------------------------------------------------------- build
    @property
    def encoder(self):
        """Sentence-embedding model, loaded on first use."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBED_MODEL, device="cpu")
        return self._encoder

    def _embed_corpus(self) -> np.ndarray:
        """Embed every recipe's ingredient list (unit-length vectors, so dot product = cosine)."""
        docs = [r.as_document() for r in self.recipes]
        return self.encoder.encode(docs, batch_size=64, normalize_embeddings=True,
                                   show_progress_bar=True).astype(np.float32)

    def save(self, path: Path = INDEX_DIR) -> None:
        """Store the embedding matrix and recipe ID order so the index can be reloaded."""
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.emb)
        (path / "ids.json").write_text(json.dumps([r.id for r in self.recipes]))

    @classmethod
    def load(cls, recipes: list[Recipe], path: Path = INDEX_DIR) -> "Retriever":
        """Rebuild a Retriever from saved embeddings (BM25 is rebuilt, it is fast).

        Fails if the corpus changed since the index was built, since rows would no
        longer line up with recipes."""
        ids = json.loads((path / "ids.json").read_text())
        if ids != [r.id for r in recipes]:
            raise RuntimeError("index out of date; run scripts/build_index.py")
        return cls(recipes, embeddings=np.load(path / "embeddings.npy"))

    # ---------------------------------------------------------------- filters
    def passes(self, r: Recipe, q: PantryQuery) -> bool:
        """True if recipe `r` meets all hard constraints of `q` (diet, time, calories,
        cuisine, exclusions). Recipes with unknown time/calories fail those limits."""
        if q.diet and not r.diets.get(q.diet, False):
            return False
        if q.max_minutes and (r.total_minutes is None or r.total_minutes > q.max_minutes):
            return False
        if q.max_calories and (r.calories is None or r.calories > q.max_calories):
            return False
        if q.cuisine and CUISINES.get(q.cuisine, q.cuisine) not in r.cuisine_text:
            return False
        for ex in q.exclude:
            if any(ex == i or ex in i.split() or _head_match(ex, i) for i in r.ingredients):
                return False
        return True

    def mask(self, q: PantryQuery | None) -> np.ndarray:
        """Boolean array over the corpus: True where the recipe passes the filters."""
        if q is None:
            return np.ones(len(self.recipes), bool)
        return np.array([self.passes(r, q) for r in self.recipes])

    # ---------------------------------------------------------------- scorers
    def bm25_scores(self, ingredients: list[str]) -> np.ndarray:
        """BM25 score of every recipe for the given pantry."""
        return np.asarray(self.bm25.get_scores(bm25_tokens(ingredients)))

    def dense_scores(self, ingredients: list[str], text: str | None = None) -> np.ndarray:
        """Cosine similarity of every recipe to the pantry (optionally plus free `text`)."""
        query = ", ".join(ingredients)  # same format as the documents
        if text:
            query += ". " + text
        v = self.encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        return (self.emb @ v[0])

    @staticmethod
    def _rank(scores: np.ndarray, mask: np.ndarray, depth: int) -> list[int]:
        """Indices of the top `depth` recipes by score, skipping filtered-out ones."""
        # filtered recipes get -inf so they sort last and are then dropped
        s = np.where(mask, scores, -np.inf)
        order = np.argsort(-s)[:depth]
        return [int(i) for i in order if np.isfinite(s[i])]

    # ---------------------------------------------------------------- search
    def search(self, q: PantryQuery, k: int = 10, mode: str = "hybrid_rerank",
               use_filters: bool = True, depth: int = 100, rrf_k: int = 60,
               w_cov: float = 0.4, w_use: float = 0.4, miss_pen: float = 0.0) -> list[Hit]:
        """mode in {bm25, dense, hybrid, hybrid_rerank}.

        hybrid_rerank score = (1-w_cov-w_use) * RRF/max(RRF) + w_cov * coverage
                              + w_use * pantry_use - miss_pen * |missing|"""
        m = self.mask(q) if use_filters else self.mask(None)
        lists: dict[str, list[int]] = {}
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            lists["bm25"] = self._rank(self.bm25_scores(q.ingredients), m, depth)
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            lists["dense"] = self._rank(self.dense_scores(q.ingredients), m, depth)

        # Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (rrf_k + rank_d).
        # It uses only ranks, so BM25 and cosine scores (different scales) need no
        # calibration; rrf_k = 60 (standard value) damps the gap between top ranks.
        fused: dict[int, float] = {}
        ranks: dict[int, dict] = {}
        for name, lst in lists.items():
            for rank, idx in enumerate(lst):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
                ranks.setdefault(idx, {})[name] = rank + 1

        hits = []
        # RRF values are tiny (~1/60); divide by the max to put them on [0, 1] like
        # coverage and pantry_use before mixing
        max_f = max(fused.values()) if fused else 1.0
        for idx, f in fused.items():
            r = self.recipes[idx]
            matched, missing, cov = coverage_stats(r, q.ingredients)
            score = f
            if mode == "hybrid_rerank":
                # blend relevance with pantry fit: coverage rewards recipes the user can
                # cook now, pantry_use rewards recipes that use up what they have
                use = pantry_use(r, q.ingredients)
                score = ((1 - w_cov - w_use) * (f / max_f) + w_cov * cov + w_use * use
                         - miss_pen * len(missing))
            hits.append(Hit(r, score, ranks.get(idx, {}), matched, missing, cov))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]
