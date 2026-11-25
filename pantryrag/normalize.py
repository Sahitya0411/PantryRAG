"""D1: ingredient normalization.

Maps free-text surface forms ("2 cups paneer cubes", "chopped capsicum") to the
canonical ingredient vocabulary built from the RecipeDB corpus.

Pipeline (each stage can be switched off for the ablation in
scripts/eval_normalization.py):
  1. clean     - lowercase, strip quantities/units/punctuation/prep descriptors
  2. singular  - naive English singularisation
  3. synonym   - longest-match lookup in a hand-built synonym map (Indian/UK/US)
  4. exact     - exact match against the vocabulary
  5. fuzzy     - RapidFuzz WRatio against the vocabulary (+ head-noun backoff)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from rapidfuzz import fuzz, process

from .lexicon import DESCRIPTORS, SYNONYMS, UNITS

_NUM = re.compile(r"\d+([./]\d+)?")
_PUNCT = re.compile(r"[^a-z\s\-']")
_DESC = {d.rstrip("_") for d in DESCRIPTORS if not d.endswith("_")}

# words where trailing 's' is not a plural
_NO_SINGULAR = {"asparagus", "madras", "molasses_", "hummus", "couscous", "molasses", "swiss", "brussels", "citrus",
                "lemongrass", "grass", "watercress", "cress", "bass", "anise", "harissa",
                "chickpeas_", "peas_", "oats", "grits", "series", "species", "chips_", "greens",
                "lentils_", "noodles_", "jus", "octopus", "chorizos_", "cheese", "rice"}


def singularize(word: str) -> str:
    if len(word) <= 3 or word in _NO_SINGULAR or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("oes", "ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith("ves") and word not in {"chives", "olives", "cloves"}:
        return word[:-3] + "f"
    if word.endswith("s"):
        return word[:-1]
    return word


def clean(text: str, drop_descriptors: bool = True) -> str:
    t = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    t = t.replace("&", " and ")
    t = re.sub(r"\(.*?\)", " ", t)
    t = _NUM.sub(" ", t)
    t = _PUNCT.sub(" ", t.replace("-", " "))
    toks = [w for w in t.split() if w not in UNITS]
    if drop_descriptors:
        toks = [w for w in toks if w not in _DESC]
    return " ".join(toks)


def canonical_form(name: str) -> str:
    """Canonical key for a corpus ingredient name (clean + singularise)."""
    words = [singularize(w) for w in clean(name, drop_descriptors=True).split()]
    return " ".join(words) or " ".join(singularize(w) for w in clean(name, False).split())


@dataclass
class NormResult:
    surface: str
    canonical: str | None
    method: str  # exact | synonym | fuzzy | head | none
    score: float = 100.0


class Normalizer:
    def __init__(self, vocab: list[str], freq: dict[str, int] | None = None,
                 use_clean=True, use_singular=True, use_synonyms=True, use_fuzzy=True,
                 fuzzy_threshold: float = 86.0):
        self.vocab = sorted(set(vocab))
        self.vocab_set = set(self.vocab)
        self.freq = freq or {}
        self.use_clean, self.use_singular = use_clean, use_singular
        self.use_synonyms, self.use_fuzzy = use_synonyms, use_fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        # synonym keys are themselves cleaned+singularised so lookup is symmetric
        self.synonyms = {}
        for k, v in SYNONYMS.items():
            self.synonyms[self._prep(k)] = canonical_form(v)
        self._max_syn_len = max(len(k.split()) for k in self.synonyms)

    def _prep(self, text: str) -> str:
        t = clean(text) if self.use_clean else text.lower().strip()
        if self.use_singular:
            t = " ".join(singularize(w) for w in t.split())
        return t

    def _resolve_target(self, name: str) -> str | None:
        """A synonym target may not be in this corpus' vocab; map it if possible."""
        if name in self.vocab_set:
            return name
        head = name.split()[-1]  # 'bell pepper' -> 'pepper' when the corpus says just "pepper"
        if head != name and self.freq.get(head, 0) >= 20:
            return head
        if self.use_fuzzy:
            cands = process.extract(name, self.vocab, scorer=fuzz.WRatio, limit=5, score_cutoff=90)
            if cands:  # prefer a frequent entry over rare noise among near-ties
                return max(cands, key=lambda c: (c[1] >= cands[0][1] - 3, self.freq.get(c[0], 0)))[0]
        return None

    @lru_cache(maxsize=50_000)
    def normalize(self, surface: str) -> NormResult:
        t = self._prep(surface)
        if not t:
            return NormResult(surface, None, "none", 0)
        if t in self.vocab_set and not (self.use_synonyms and t in self.synonyms
                                        and self.synonyms[t] in self.vocab_set):
            return NormResult(surface, t, "exact")
        if self.use_synonyms:
            if t in self.synonyms:
                tgt = self._resolve_target(self.synonyms[t])
                if tgt:
                    return NormResult(surface, tgt, "synonym")
            # longest synonym n-gram inside the phrase ("fresh palak leaves")
            words = t.split()
            for n in range(min(self._max_syn_len, len(words)), 0, -1):
                for i in range(len(words) - n + 1):
                    g = " ".join(words[i:i + n])
                    if g in self.synonyms:
                        rest = " ".join(words[:i] + [self.synonyms[g]] + words[i + n:])
                        for cand in (rest, self.synonyms[g]):
                            tgt = cand if cand in self.vocab_set else None
                            if tgt:
                                return NormResult(surface, tgt, "synonym")
        if t in self.vocab_set:
            return NormResult(surface, t, "exact")
        if self.use_fuzzy:
            m = process.extractOne(t, self.vocab, scorer=fuzz.WRatio,
                                   score_cutoff=self.fuzzy_threshold)
            if m:
                # prefer the more frequent of near-tied candidates
                cands = process.extract(t, self.vocab, scorer=fuzz.WRatio, limit=5,
                                        score_cutoff=m[1] - 2)
                best = max(cands, key=lambda c: (c[1], self.freq.get(c[0], 0)))
                return NormResult(surface, best[0], "fuzzy", best[1])
            # head-noun backoff: "paneer tikka masala cubes" -> try last 2/1 words
            words = t.split()
            for n in (2, 1):
                if len(words) > n:
                    head = " ".join(words[-n:])
                    if head in self.vocab_set:
                        return NormResult(surface, head, "head", 80)
        return NormResult(surface, None, "none", 0)

    def normalize_many(self, items: list[str]) -> list[str]:
        out, seen = [], set()
        for it in items:
            r = self.normalize(it)
            if r.canonical and r.canonical not in seen:
                seen.add(r.canonical)
                out.append(r.canonical)
        return out


# ---------------------------------------------------------------- recipe phrases
_TAIL = re.compile(
    r",|\(|\bcut\b|\bon the\b|\bto serve\b|\bto garnish\b|\bfor\b|\bplus\b|\bor\b|\binto\b|"
    r"\blarger\b|\bat room\b|\bdivided\b|\bseparated\b|\bif\b|\bsuch as\b|\bin\b|\bwith\b|"
    r"\bleaves picked\b|\bstems\b|\bstalks? removed\b|\bkept\b|\breserved\b")


_PHRASE_DROP = {"lean", "fine", "squeeze", "removed", "core", "reduced", "dry", "good",
                "generous", "heaped", "level", "rounded", "light", "mature", "low",
                "unwaxed", "free", "range", "skin", "on", "off", "use", "the", "large"}
# trailing part-words dropped when the rest is a known ingredient ('parsley leaf' -> parsley)
_PART = {"leaf", "clove", "floret", "spear", "wedge", "bulb", "fillet", "stick", "sprig",
         "cube", "slice", "piece", "chunk", "segment", "zest", "breast_"}
_KEEP_PART = {"bay leaf", "curry leaf", "lime leaf", "kaffir lime leaf", "chicken breast",
              "stock cube", "lemon zest", "orange zest", "lime zest", "fish fillet"}


def phrase_to_name(phrase: str) -> str:
    """'500g sweet potatoes cut into wedges' -> 'sweet potato';
    '4 skinless salmon fillets' -> 'salmon'; '2 tbsp double cream' -> 'double cream'."""
    p = unicodedata.normalize("NFKD", phrase.lower()).encode("ascii", "ignore").decode()
    p = re.sub(r"\d+\s*(g|kg|ml|l|oz|lb|cm|mm)\b", " ", p)          # '500g', '5cm'
    for seg in p.split(","):                     # first segment that names something
        seg = _TAIL.split(seg, maxsplit=1)[0].split(" and ")[0]
        words = [w for w in canonical_form(seg).split() if w not in _PHRASE_DROP]
        if words and not all(w in _DESC for w in words):
            return " ".join(words)
    return canonical_form(p)


def strip_part(name: str, names: dict[str, int], head_min: int = 5) -> str:
    w = name.split()
    if len(w) > 1 and w[-1] in _PART and name not in _KEEP_PART:
        rest = " ".join(w[:-1])
        if names.get(rest, 0) >= head_min:
            return rest
    return name


def backoff_names(names: dict[str, int], min_count: int = 3, head_min: int = 5) -> dict[str, str]:
    """Map rare names to their most specific frequent suffix
    ('long stem broccoli' -> 'broccoli'); frequent names map to themselves."""
    out = {}
    for n, c in names.items():
        if c >= min_count:
            out[n] = n
            continue
        w = n.split()
        target = n
        for k in range(1, len(w)):
            suf = " ".join(w[k:])
            if names.get(suf, 0) >= head_min:
                target = suf
                break
        out[n] = target
    return out
