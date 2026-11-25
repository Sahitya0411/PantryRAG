"""D3/D4 evaluation: faithfulness of RAG answers vs. a plain-LLM baseline.

For each pantry query we generate (a) a PantryRAG answer and (b) a plain-LLM
answer with the same model and no retrieval, then score:

  ingredient_grounding  share of ingredients mentioned in the answer that are
                        backed by the reference recipe, the user's pantry, or a
                        suggested substitute. Reference recipe = the cited
                        recipe for RAG; for the plain LLM, the corpus recipe
                        with the highest ingredient overlap (generous to it).
  step_grounding        share of generated steps whose content words mostly
                        (>= 50%) appear in some step of the reference recipe.
  unflagged_missing     ingredients used in the steps that the user does not
                        have and the answer did not list as missing (per answer).
  diet_violations       answers that use an ingredient breaking the diet.
  citation_validity     (RAG) cited IDs that exist in the retrieved set.

    python scripts/eval_generation.py --backend hf --n-synthetic 15
Outputs are cached in results/generations_<backend>.jsonl (resumable).
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pantryrag.diet import recipe_diets  # noqa: E402
from pantryrag.generation import CITE_RE, plain_answer, rag_answer  # noqa: E402
from pantryrag.lexicon import STAPLES  # noqa: E402
from pantryrag.normalize import clean, singularize  # noqa: E402
from pantryrag.pipeline import PantryRAG  # noqa: E402
from pantryrag.retrieval import _head_match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STOP = set("""a an the and or of to in on for with into onto over until then it its is are be
until about mins min minutes minute heat add stir cook until well large small pan bowl
your you this that from at by as up out off all each some few more half rest remaining
season taste serve serving""".split())


# ------------------------------------------------------------------ text utils
def section(text: str, name: str) -> str:
    m = re.search(rf"\*\*{name}[^*]*\*\*:?(.*?)(?=\n\*\*[A-Z][a-zA-Z ]+:?\*\*|\Z)", text, re.S | re.I)
    return m.group(1) if m else ""


def steps_of(text: str) -> list[str]:
    body = section(text, "Steps")
    lines = re.split(r"\n\s*(?:\d+[.)]|[-*])\s*|(?<=\.)\s+(?=\d+\.\s)", "\n" + body)
    return [l.strip() for l in lines if len(l.strip().split()) >= 3]


def content_words(s: str) -> set[str]:
    return {singularize(w) for w in clean(s, drop_descriptors=False).split()
            if w not in STOP and len(w) > 2}


# vocabulary entries that are not specific ingredients (NER noise / generic words)
GENERIC = {"spoon", "sprinkle", "spice", "sauce", "salad", "spread", "peel", "seed", "leaf",
           "herb", "veg", "vegetable", "toast", "nut", "cheese", "cream", "juice", "zest",
           "oil", "stock", "fruit", "meat", "fish", "flour", "sugar", "dressing", "topping",
           "filling", "mix", "paste", "powder", "pastry", "dough", "glaze", "garnish", "side",
           "dip", "crumb", "liquid", "rind", "skin", "bone", "tin", "bunch", "handful", "drizzle",
           "dash", "pinch", "splash", "knob", "dollop", "piece", "slice", "chunk", "cube", "wedge",
           "serve", "season", "seasoning", "mixture", "batter", "marinade", "pan", "dish"}


class IngredientExtractor:
    def __init__(self, vocab: dict[str, int], min_count: int = 5):
        self.v = {k for k, c in vocab.items() if c >= min_count and len(k) > 2 and k not in GENERIC}

    def __call__(self, text: str) -> list[str]:
        toks = [singularize(w) for w in clean(text, drop_descriptors=False).split()]
        out, i = [], 0
        while i < len(toks):
            for n in (3, 2, 1):
                g = " ".join(toks[i:i + n])
                if len(toks[i:i + n]) == n and g in self.v:
                    out.append(g)
                    i += n
                    break
            else:
                i += 1
        return list(dict.fromkeys(out))


def nonstaple(i: str) -> bool:
    return i not in STAPLES and not any(_head_match(i, s) for s in STAPLES)


def backed(i: str, pool: list[str]) -> bool:
    return any(_head_match(i, p) or i in p.split() or p in i.split() for p in pool)


# ------------------------------------------------------------------ queries
def build_queries(rag: PantryRAG, n_syn: int, seed: int = 7):
    qs = [json.loads(l)["text"] for l in open(ROOT / "data/eval/pantry_queries_gold.jsonl")]
    rng = random.Random(seed)
    pool = [r for r in rag.recipes if len([i for i in r.ingredients if nonstaple(i)]) >= 5]
    for r in rng.sample(pool, n_syn):
        ings = rng.sample([i for i in r.ingredients if nonstaple(i)], rng.randint(3, 5))
        extra = rng.choice(["", "; vegetarian", "; under 40 minutes", "", "; quick"])
        qs.append(f"I have {', '.join(ings[:-1])} and {ings[-1]}{extra}")
    return qs


# ------------------------------------------------------------------ scoring
def score(rec: dict, rag: PantryRAG, extract: IngredientExtractor) -> dict:
    text = rec["text"]
    pantry = rec["pantry"]
    body = re.split(r"\*\*Alternatives", text)[0]
    mentioned = [i for i in extract(body) if nonstaple(i)]
    if rec["system"] == "rag":
        ids = rec["cited_ids"] or rec["retrieved_ids"][:1]
        ref = rag.recipes[rag.retriever.id2idx[ids[0]]] if ids else None
    else:  # best-matching real recipe for the plain answer
        ms = set(mentioned)
        best, best_j = None, -1
        for r in rag.recipes:
            s = set(r.ingredients)
            j = len(ms & s) / (len(ms | s) or 1)
            if j > best_j:
                best, best_j = r, j
        ref = best
    ref_ings = ref.ingredients if ref else []
    # recipes often mention items in their steps that are missing from their
    # ingredient list (e.g. "serve with rice"); those are still in the document
    ref_doc = ref_ings + (extract(" ".join(ref.steps)) if ref else [])
    subs = rec.get("substitutes", [])
    pool = ref_doc + pantry + subs
    grounded = [i for i in mentioned if backed(i, pool)]
    gsteps = steps_of(text)
    ref_steps = [content_words(s) for s in (ref.steps if ref else [])]
    step_ok = 0
    for s in gsteps:
        cw = content_words(s)
        if cw and ref_steps and max(len(cw & r) / len(cw) for r in ref_steps) >= 0.5:
            step_ok += 1
    missing_sec = extract(section(text, "Missing"))
    step_ings = [i for i in extract(" ".join(gsteps)) if nonstaple(i)]
    unflagged = [i for i in step_ings if not backed(i, pantry) and not backed(i, missing_sec)
                 and not backed(i, subs)]
    diet = rec.get("diet")
    diet_bad = bool(diet) and not recipe_diets(mentioned).get(diet, True)
    out = {
        "ingredient_grounding": len(grounded) / len(mentioned) if mentioned else None,
        "step_grounding": step_ok / len(gsteps) if gsteps else None,
        "n_steps": len(gsteps),
        "unflagged_missing": len(unflagged),
        "diet_violation": diet_bad if diet else None,
        "ref_recipe": ref.id if ref else None,
    }
    if rec["system"] == "rag":
        cited = CITE_RE.findall(text) + ["x"] * len(rec.get("invalid", []))
        out["citation_validity"] = (len(rec["cited_ids"]) / len(cited)) if cited else 0.0
        out["has_citation"] = bool(rec["cited_ids"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="hf")
    ap.add_argument("--n-synthetic", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries")
    args = ap.parse_args()
    rag = PantryRAG(llm=args.backend)
    params = json.loads((ROOT / "results/rerank_params.json").read_text()) \
        if (ROOT / "results/rerank_params.json").exists() else {}
    out_path = ROOT / f"results/generations_{args.backend}.jsonl"
    done = {(r["query"], r["system"]) for r in map(json.loads, out_path.read_text().splitlines())} \
        if out_path.exists() else set()
    queries = build_queries(rag, args.n_synthetic)[:args.limit]
    with open(out_path, "a") as f:
        for qi, text in enumerate(queries):
            q = rag.parse(text)
            for system in ("rag", "plain"):
                if (text, system) in done:
                    continue
                t0 = time.time()
                if system == "rag":
                    hits = rag.retriever.search(q, k=10, **params)
                    a = rag_answer(q, hits, rag.llm, rag.substituter)
                    subs = [s.substitute for v in a.subs.values() for s in v]
                    rec = {"retrieved_ids": [h.recipe.id for h in a.hits], "cited_ids": a.cited_ids,
                           "invalid": a.invalid_citations, "substitutes": subs}
                else:
                    a = plain_answer(q, rag.llm)
                    rec = {"retrieved_ids": [], "cited_ids": [], "invalid": [], "substitutes": []}
                rec.update({"query": text, "system": system, "backend": a.backend, "text": a.text,
                            "pantry": q.ingredients, "diet": q.diet, "seconds": round(time.time() - t0, 1)})
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[{qi+1}/{len(queries)}] {system} {rec['seconds']}s", flush=True)

    recs = [json.loads(l) for l in out_path.read_text().splitlines()]
    extract = IngredientExtractor(rag.vocab)
    agg = {}
    for system in ("rag", "plain"):
        rows = [score(r, rag, extract) for r in recs if r["system"] == system]
        def mean(key):
            v = [r[key] for r in rows if r.get(key) is not None]
            return round(sum(v) / len(v), 3) if v else None
        agg[system] = {k: mean(k) for k in ("ingredient_grounding", "step_grounding", "n_steps",
                                             "unflagged_missing", "diet_violation",
                                             "citation_validity", "has_citation")}
        agg[system]["n"] = len(rows)
    agg["backend"] = recs[0]["backend"] if recs else args.backend
    print(json.dumps(agg, indent=2))
    (ROOT / f"results/generation_{args.backend}.json").write_text(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
