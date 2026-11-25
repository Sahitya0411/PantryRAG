"""End-to-end PantryRAG pipeline + multi-turn dialogue state."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from . import corpus
from .generation import Answer, rag_answer
from .llm import LLM, get_llm
from .normalize import Normalizer
from .pantry_parser import PantryQuery, parse_constraints, parse_pantry, split_items
from .retrieval import Retriever
from .substitutions import Substituter


class PantryRAG:
    """Single-turn system: loads the corpus, normaliser, retriever and substituter
    once, and answers a pantry message end to end.

    `llm` is a backend instance or a backend name (resolved lazily on first use)."""

    def __init__(self, llm: LLM | str | None = None, min_vocab_count: int = 2):
        self.recipes, self.vocab = corpus.load()
        # ingredients seen only once are mostly noise; keep them out of the normaliser
        names = [k for k, c in self.vocab.items() if c >= min_vocab_count]
        self.normalizer = Normalizer(names, self.vocab)
        self.retriever = Retriever.load(self.recipes)
        self.substituter = Substituter(self.recipes, encoder=self.retriever.encoder)
        self._llm = llm

    @property
    def llm(self) -> LLM:
        """The generation backend, created on first access (loading a model is slow)."""
        if self._llm is None or isinstance(self._llm, str):
            self._llm = get_llm(self._llm)
        return self._llm

    def parse(self, text: str) -> PantryQuery:
        """Parse a pantry message into ingredients + constraints."""
        return parse_pantry(text, self.normalizer)

    def answer(self, text: str, k: int = 10, mode: str = "hybrid_rerank") -> tuple[PantryQuery, Answer]:
        """Parse -> retrieve top `k` recipes with `mode` -> generate. Returns (query, answer)."""
        q = self.parse(text)
        hits = self.retriever.search(q, k=k, mode=mode)
        return q, rag_answer(q, hits, self.llm, self.substituter)


# ---------------------------------------------------------------- multi-turn
# Follow-up intents. Group 1 of ADD_RE / REMOVE_RE is the list of ingredients after
# the cue phrase ("I also have peas and corn" -> " peas and corn").
ADD_RE = re.compile(r"\b(?:i (?:also )?have|i've (?:also )?got|add|also|plus|and i have|i found)\b(.*)", re.I)
REMOVE_RE = re.compile(r"\b(?:i (?:don'?t|do not) have|i'?m out of|ran out of|no more|"
                       r"remove|i used up)\b(.*)", re.I)
# "show me the second one", "let's make #1", "go with the last": group 1 is the
# ordinal, mapped to a list index through ORD below.
PICK_RE = re.compile(r"\b(?:(?:show|tell|give)(?: me)?(?: more)?(?: about| the steps for| steps for)?|"
                     r"steps for|more about|let'?s (?:do|make|go with)|i'?ll (?:make|try)|"
                     r"(?:pick|choose|go with))\s+(?:the\s+)?(?:option\s*|#|number\s*|recipe\s*)?"
                     r"(\d|first|second|third|1st|2nd|3rd|last|that one)\b", re.I)
# "what can I use instead of cream?": group 1 is the ingredient to replace.
SUB_RE = re.compile(r"(?:instead of|substitute(?: for)?|replace|swap(?: out)?|without)\s+(?:the\s+)?([a-z ]+?)\s*(?:\?|$|with|in)", re.I)
RESET_RE = re.compile(r"\b(start over|reset|new pantry|clear)\b", re.I)
# ordinal word -> index into the current hit list (-1 = last)
ORD = {"1": 0, "first": 0, "1st": 0, "2": 1, "second": 1, "2nd": 1, "3": 2, "third": 2,
       "3rd": 2, "last": -1, "that one": 0}


@dataclass
class DialogueState:
    """What the chat remembers between turns: current query, last retrieved hits,
    and the message history passed to the LLM."""
    query: PantryQuery | None = None
    hits: list = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


class ChatSession:
    """Keeps pantry + constraints across turns so follow-ups can refine them.

    Supported follow-ups: "I also have potatoes", "I don't have spinach",
    "make it vegan" / "under 20 minutes" / "no onion", "show me the second one",
    "what can I use instead of cream?", "start over".
    """

    def __init__(self, rag: PantryRAG):
        self.rag = rag
        self.state = DialogueState()

    def _merge_constraints(self, text: str) -> bool:
        """Add constraints found in `text` ("make it vegan", "no onion") to the current
        query. Returns True if anything changed."""
        cons, _ = parse_constraints(text, self.rag.normalizer)
        if not cons or self.state.query is None:
            return False
        q = self.state.query
        # exclusions accumulate across turns; other constraints overwrite old values
        excl = list(dict.fromkeys(q.exclude + cons.pop("exclude", [])))
        self.state.query = replace(q, exclude=excl, **cons)
        self.state.query.ingredients = [i for i in q.ingredients if i not in excl]
        return True

    def _items(self, text: str) -> list[str]:
        """Canonical ingredient names mentioned in a fragment of text."""
        return self.rag.normalizer.normalize_many(split_items(text.lower()))

    def turn(self, text: str) -> tuple[str, Answer | None]:
        """Handle one user message and return (reply text, Answer or None).

        Intents are checked in priority order: reset / first message, pick a listed
        recipe, ask for a substitute, new pantry, then add/remove ingredients or
        change constraints. Anything else is answered as a question about the
        current results."""
        s = self.state
        # 1) first message or explicit reset: start a fresh state
        if RESET_RE.search(text) or s.query is None:
            s.__init__()
            # a bare "start over" has no pantry in it, so just prompt for one
            if RESET_RE.search(text) and len(text.split()) <= 3:
                return "Okay, starting fresh. What's in your pantry?", None
            s.query = self.rag.parse(text)
            return self._retrieve_and_answer(text)

        # 2) "show me the second one": move that hit to the front and re-answer
        # without retrieving again
        m = PICK_RE.search(text)
        if m and s.hits:
            i = ORD.get(m.group(1).lower(), 0)
            # clamp out-of-range picks to the last hit
            i = min(i, len(s.hits) - 1) if i >= 0 else len(s.hits) - 1
            h = s.hits.pop(i)
            s.hits.insert(0, h)  # chosen recipe becomes the focus
            return self._answer(text)

        # 3) substitution question about the focused recipe: answered directly from
        # the Substituter (no LLM call). Skipped when the message also adds items,
        # since "add X instead of Y" is an edit, not a question.
        m = SUB_RE.search(text)
        if m and s.hits and not ADD_RE.search(text):
            ing = self.rag.normalizer.normalize(m.group(1)).canonical or m.group(1).strip()
            subs = self.rag.substituter.suggest(ing, s.query.ingredients, s.query.diet, k=3)
            r = s.hits[0].recipe
            if not subs:
                msg = f"I don't have a reliable substitute for **{ing}** in [R{r.id}] {r.title}."
            else:
                lines = [f"- **{x.substitute}**{' (you have it)' if x.in_pantry else ''}: {x.note}" for x in subs]
                msg = f"Substitutes for **{ing}** in {r.title} [R{r.id}]:\n" + "\n".join(lines)
            s.history += [{"role": "user", "content": text}, {"role": "assistant", "content": msg}]
            return msg, None

        # a message that lists 3+ ingredients with no add/remove wording is a NEW
        # pantry: start over instead of merging into the old constraints
        if not (ADD_RE.search(text) or REMOVE_RE.search(text)):
            q2 = self.rag.parse(text)
            if len(q2.ingredients) >= 3:
                s.query = q2
                s.history = []
                return self._retrieve_and_answer(text)

        # 4) incremental edits: remove or add ingredients, then merge any constraints
        changed = False
        m = REMOVE_RE.search(text)
        if m:
            gone = set(self._items(m.group(1)))
            s.query.ingredients = [i for i in s.query.ingredients if i not in gone]
            changed = True
        else:
            m = ADD_RE.search(text)
            if m:
                for i in self._items(m.group(1)):
                    if i not in s.query.ingredients:
                        s.query.ingredients.append(i)
                changed = True
        changed = self._merge_constraints(text) or changed
        if not changed:
            # treat as a brand-new pantry if it contains ingredients, else a question
            q2 = self.rag.parse(text)
            if len(q2.ingredients) >= 2:
                s.query = q2
                changed = True
        # 5) the pantry changed -> retrieve again; otherwise answer from the same hits
        if changed:
            return self._retrieve_and_answer(text)
        return self._answer(text)

    def _retrieve_and_answer(self, text: str):
        """Re-run retrieval for the current query, then generate an answer."""
        s = self.state
        s.hits = self.rag.retriever.search(s.query, k=10)
        return self._answer(text)

    def _answer(self, text: str):
        """Generate an answer from the current hits and record the exchange in history."""
        s = self.state
        ans = rag_answer(s.query, s.hits, self.rag.llm, self.rag.substituter,
                         history=s.history, user_text=text)
        s.history += [{"role": "user", "content": text}, {"role": "assistant", "content": ans.text}]
        return ans.text, ans
