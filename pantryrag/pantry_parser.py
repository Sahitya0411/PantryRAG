"""D1: parse a free-text pantry message into ingredients + hard constraints.

    "I have paneer, spinach, rice, onion and tomato; vegetarian, under 30 minutes"
 -> PantryQuery(ingredients=[paneer, spinach, rice, onion, tomato],
                diet="vegetarian", max_minutes=30)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .normalize import Normalizer

CUISINES = {
    # user keyword -> substring matched against "region | sub_region | continent" (lowercase)
    "indian": "indian", "south indian": "indian", "punjabi": "indian", "pakistani": "pakistani",
    "bangladeshi": "bangladeshi", "sri lankan": "sri lankan", "nepali": "nepalese",
    "italian": "italian", "mexican": "mexican", "chinese": "chinese", "thai": "thai",
    "japanese": "japanese", "korean": "korean", "french": "french", "greek": "greek",
    "spanish": "spanish", "middle eastern": "middle eastern", "lebanese": "lebanese",
    "turkish": "turkish", "american": "us", "british": "uk", "irish": "irish",
    "caribbean": "caribbean", "african": "african", "vietnamese": "vietnamese",
    "indonesian": "indonesian", "filipino": "filipino", "german": "deutschland",
    "moroccan": "moroccan", "asian": "asian", "european": "european", "mediterranean": "greek",
    "canadian": "canadian", "australian": "australian", "russian": "russian",
    "scandinavian": "scandinavian", "south american": "south american", "brazilian": "brazilian",
}

DIET_PATTERNS = [
    ("vegan", r"\bvegan\b|plant[- ]based|no animal products?"),
    ("eggless_vegetarian", r"\beggless\b|\bjain\b|vegetarian,? no eggs?|no eggs?.{0,20}vegetarian"),
    ("vegetarian", r"\bvegetarian\b|\bveggie\b|\bveg\b(?!etable)|\bno meat\b|meat[- ]free|meatless"),
    ("pescatarian", r"\bpescatarian\b|\bpescetarian\b"),
    ("dairy_free", r"dairy[- ]free|no dairy|lactose[- ]intolerant|without dairy"),
]
EXCLUDE_RE = re.compile(
    r"\b(?:no|without|avoid|avoiding|exclude|excluding|minus|not|allergic to|hate|don'?t like|"
    r"don'?t want|skip)\s+([a-z][a-z \-]{1,30}?)(?=[,.;!?]|\band\b|\bor\b|$)")
TIME_RE = re.compile(
    r"(?:under|less than|within|in|max(?:imum)?|at most|<|below|no more than)\s*(\d+(?:\.\d+)?)\s*"
    r"(min(?:ute)?s?|h(?:ou)?rs?|hours?)\b")
CAL_RE = re.compile(r"(?:under|less than|below|max(?:imum)?|<|at most)\s*(\d{2,4})\s*(?:k?cal|calories)")
LEAD_IN = re.compile(
    r"^(?:hi|hey|hello)?[ ,]*(?:i(?:'ve| have)?(?: only)?(?: got)?|i've got|we have|there(?:'s| is| are)|"
    r"my (?:pantry|fridge) has|what can i (?:make|cook) with|suggest (?:a )?(?:recipe|something) with|"
    r"(?:a )?recipe (?:using|with)|using|with|got|have|only|just|also|plus)\b[: ,]*")
NOISE_CHUNKS = {"", "please", "thanks", "thank you", "something", "anything", "ideas", "quick",
                "dinner", "lunch", "breakfast", "snack", "recipe", "recipes", "help", "easy",
                "healthy", "tonight", "today"}


@dataclass
class PantryQuery:
    text: str
    ingredients: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    diet: str | None = None
    exclude: list[str] = field(default_factory=list)
    max_minutes: float | None = None
    max_calories: float | None = None
    cuisine: str | None = None

    def constraints_text(self) -> str:
        parts = []
        if self.diet:
            parts.append(self.diet.replace("_", " "))
        if self.max_minutes:
            parts.append(f"<= {int(self.max_minutes)} min")
        if self.max_calories:
            parts.append(f"<= {int(self.max_calories)} kcal/serving")
        if self.cuisine:
            parts.append(self.cuisine)
        if self.exclude:
            parts.append("no " + ", ".join(self.exclude))
        return "; ".join(parts) or "none"


def parse_constraints(text: str, norm: Normalizer) -> tuple[dict, str]:
    """Return constraint dict and the text with constraint spans removed."""
    t = " " + text.lower() + " "
    c: dict = {}
    for diet, pat in DIET_PATTERNS:
        if re.search(pat, t):
            c["diet"] = diet
            t = re.sub(pat, " , ", t)
            break
    m = TIME_RE.search(t)
    if m:
        v = float(m.group(1))
        c["max_minutes"] = v * 60 if m.group(2).startswith("h") else v
        t = t[:m.start()] + " , " + t[m.end():]
    elif re.search(r"\b(quick|fast|in a hurry|short on time)\b", t):
        c["max_minutes"] = 30
    m = CAL_RE.search(t)
    if m:
        c["max_calories"] = float(m.group(1))
        t = t[:m.start()] + " , " + t[m.end():]
    for kw in sorted(CUISINES, key=len, reverse=True):
        if re.search(rf"\b{kw}\b", t):
            c["cuisine"] = kw
            t = re.sub(rf"\b{kw}\b(?:\s+(?:food|style|cuisine|dish))?", " , ", t)
            break
    excl = []
    for m in EXCLUDE_RE.finditer(t):
        phrase = m.group(1).strip()
        if phrase in {"time", "oven", "more", "much", "too", "sure"}:
            continue
        r = norm.normalize(phrase)
        excl.append(r.canonical or phrase)
    if excl:
        c["exclude"] = excl
        t = EXCLUDE_RE.sub(" , ", t)
    t = re.sub(r"\b(minutes?|mins?|calories|please|thanks|make|cook|for|dinner|lunch|"
               r"breakfast|tonight|something|anything|want|would like|like|can|you|suggest|"
               r"what|recipe|recipes|dish|meal|ideas?|i)\b", " ", t)
    return c, t


def split_items(text: str) -> list[str]:
    parts = re.split(r"[,;\n/+]|\.(?=\s)|\band\b|&|\bplus\b|\balso\b", text)
    out = []
    for p in parts:
        p = LEAD_IN.sub("", p.strip()).strip(" .!?:-")
        if p and p not in NOISE_CHUNKS:
            out.append(p)
    return out


def parse_pantry(text: str, norm: Normalizer) -> PantryQuery:
    cons, rest = parse_constraints(text, norm)
    q = PantryQuery(text=text, **cons)
    seen = set()
    for chunk in split_items(rest):
        r = norm.normalize(chunk)
        found = [r.canonical] if r.canonical else []
        if not found:  # scan n-grams inside the chunk ("leftover rice from yesterday")
            words = chunk.split()
            for n in (3, 2, 1):
                for i in range(len(words) - n + 1):
                    g = " ".join(words[i:i + n])
                    rr = norm.normalize(g)
                    if rr.canonical and rr.method in {"exact", "synonym"}:
                        found.append(rr.canonical)
                if found:
                    break
        if not found:
            q.unresolved.append(chunk)
        for f in found:
            if f not in seen and f not in q.exclude:
                seen.add(f)
                q.ingredients.append(f)
    return q
