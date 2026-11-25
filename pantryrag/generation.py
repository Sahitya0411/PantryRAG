"""D3: citation-grounded generation (and the plain-LLM baseline)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLM, ExtractiveLLM
from .pantry_parser import PantryQuery
from .retrieval import Hit
from .substitutions import Substituter, Substitution

RAG_SYSTEM = """You are PantryRAG, a cooking assistant that recommends recipes from a recipe database.

Rules (follow strictly):
1. Only recommend recipes that appear in RETRIEVED RECIPES. Never invent a recipe.
2. Cite every recipe you mention with its ID in square brackets, e.g. [R4255].
3. Ingredients and steps must come from the cited recipe. You may shorten or rephrase steps, and you may apply a listed substitution, but do not add new ingredients or techniques.
4. For the recommended recipe, list the MISSING ingredients exactly as given, each with the suggested substitute if one is listed.
5. Respect the user's constraints. If no retrieved recipe fits well, say so honestly.

Answer format (markdown):
**Recommendation:** <title> [R<id>]
**Why:** one sentence (pantry coverage, constraints).
**You have:** ...
**Missing:** ... (with substitutes)
**Steps:** numbered, concise, from the cited recipe.
**Alternatives:** up to 2 other retrieved recipes, one line each, with citations."""

PLAIN_SYSTEM = """You are a helpful cooking assistant. Suggest one recipe the user can cook with the ingredients they have, respecting any constraints.

Answer format (markdown):
**Recommendation:** <title>
**Why:** one sentence.
**You have:** ...
**Missing:** ingredients the user needs to buy (with substitutes)
**Steps:** numbered, concise.
**Alternatives:** up to 2 other dish ideas, one line each."""

CITE_RE = re.compile(r"\[R([A-Za-z0-9]+)\]")


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    subs: dict[str, list[Substitution]] = field(default_factory=dict)
    cited_ids: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    backend: str = ""


def substitutions_for(hit: Hit, q: PantryQuery, sub: Substituter, k: int = 2):
    return {m: sub.suggest(m, q.ingredients, q.diet, k=k) for m in hit.missing}


def build_context(q: PantryQuery, hits: list[Hit], subs_by_id: dict[str, dict]) -> str:
    blocks = []
    for h in hits:
        subs = subs_by_id.get(h.recipe.id, {})
        miss = []
        for m in h.missing:
            s = subs.get(m) or []
            if s:
                alts = "; ".join(f"{x.substitute}{' (in pantry)' if x.in_pantry else ''} - {x.note}" for x in s)
                miss.append(f"{m} -> substitute: {alts}")
            else:
                miss.append(f"{m} (no substitute known)")
        blocks.append(
            f"{h.recipe.as_context()}\n"
            f"User has: {', '.join(h.matched) or 'none'} (coverage {h.coverage:.0%})\n"
            f"MISSING: " + ("\n  - " + "\n  - ".join(miss) if miss else "nothing") )
    return ("RETRIEVED RECIPES (ranked best first):\n\n" + "\n\n".join(blocks) +
            f"\n\nUSER PANTRY: {', '.join(q.ingredients) or 'unknown'}"
            f"\nUSER CONSTRAINTS: {q.constraints_text()}")


def extractive_answer(q: PantryQuery, hits: list[Hit], subs_by_id: dict) -> str:
    if not hits:
        return ("I couldn't find a recipe in the database that matches your pantry and "
                f"constraints ({q.constraints_text()}). Try relaxing a constraint.")
    h = hits[0]
    r = h.recipe
    subs = subs_by_id.get(r.id, {})
    miss = []
    for m in h.missing:
        s = subs.get(m) or []
        miss.append(f"{m}" + (f" (or {s[0].substitute}{' - you have it' if s[0].in_pantry else ''})" if s else ""))
    steps = "\n".join(f"{i+1}. {s[0].upper() + s[1:]}" for i, s in enumerate(r.steps[:10]))
    alts = "\n".join(f"- {x.recipe.title} [R{x.recipe.id}] - you have {x.coverage:.0%} of it"
                     for x in hits[1:3])
    return (f"**Recommendation:** {r.title} [R{r.id}]\n"
            f"**Why:** you already have {h.coverage:.0%} of the non-staple ingredients"
            f"{' and it fits: ' + q.constraints_text() if q.constraints_text() != 'none' else ''}.\n"
            f"**You have:** {', '.join(h.matched) or '-'}\n"
            f"**Missing:** {', '.join(miss) or 'nothing'}\n"
            f"**Steps:**\n{steps}\n"
            f"**Alternatives:**\n{alts or '-'}")


def check_citations(text: str, hits: list[Hit]) -> tuple[list[str], list[str]]:
    valid = {h.recipe.id for h in hits}
    cited = list(dict.fromkeys(CITE_RE.findall(text)))
    return [c for c in cited if c in valid], [c for c in cited if c not in valid]


def rag_answer(q: PantryQuery, hits: list[Hit], llm: LLM, sub: Substituter,
               history: list[dict] | None = None, user_text: str | None = None,
               n_context: int = 3) -> Answer:
    hits = hits[:n_context]
    subs_by_id = {h.recipe.id: substitutions_for(h, q, sub) for h in hits}
    if isinstance(llm, ExtractiveLLM) or not hits:
        text = extractive_answer(q, hits, subs_by_id)
    else:
        ctx = build_context(q, hits, subs_by_id)
        msgs = list(history or [])[-4:]
        msgs.append({"role": "user", "content": f"{ctx}\n\nUSER MESSAGE: {user_text or q.text}"})
        text = llm.chat(RAG_SYSTEM, msgs)
    valid, invalid = check_citations(text, hits)
    for bad in invalid:  # never show a citation we can't back
        text = text.replace(f"[R{bad}]", "[citation removed: not in retrieved set]")
    return Answer(text, hits, subs_by_id.get(hits[0].recipe.id, {}) if hits else {},
                  valid, invalid, getattr(llm, "name", "?"))


def plain_answer(q: PantryQuery, llm: LLM) -> Answer:
    text = llm.chat(PLAIN_SYSTEM, [{"role": "user", "content": q.text}])
    return Answer(text, [], {}, [], CITE_RE.findall(text), getattr(llm, "name", "?"))
