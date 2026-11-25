# PantryRAG: method notes (feeds the final report)

## D1. Data pipeline and ingredient normalization

**Corpus.** RecipeDB 2.0 via `recipe2-api`. Metadata comes from `recipe/recipesinfo` (paged, at most 10
per page) and `recipes_cuisine/cuisine/{region}`, ingredients from `recipe-nutri/nutritioninfo` (one row
per ingredient with USDA-mapped nutrients), and steps from `instructions/{id}`. The corpus mixes random
pages across all regions with the first N pages of *Indian Subcontinent*, so that Indian pantries are well
covered. Records with fewer than 3 ingredients or no steps are dropped, and exact (title, ingredient-set)
duplicates are removed.

**Canonical vocabulary.** Each corpus ingredient name is lowercased, accent-stripped, cleaned of
quantities, units and prep descriptors ("chopped", "fresh", "fillet", ...) and singularised. Unambiguous
regional synonyms are folded (e.g. *aubergine → eggplant*). The canonical set with frequencies is the
vocabulary.

**Normalizer** (surface form → vocabulary entry), in this order:
1. cleaning (quantities, units, descriptors, punctuation, accents),
2. singularisation,
3. a hand-built synonym map of about 230 entries covering Hindi/Hinglish (*jeera, haldi, palak, dahi,
   atta, maida, besan, rajma …*), British/US (*courgette, double cream, minced beef, coriander leaves
   …*) and shorthands. Synonyms are matched on the whole phrase, then on the longest n-gram inside it,
4. an exact vocabulary match,
5. a fuzzy match (RapidFuzz WRatio ≥ 86, ties broken by corpus frequency), then backing off to the
   head noun.

**Pantry parser.** Constraints are pulled out first with regex patterns: diet (vegan, eggless,
vegetarian, pescatarian, dairy-free), time ("under 30 minutes", "in 1 hour", "quick" means 30), calories,
cuisine (keyword → region substring) and exclusions ("no X", "without X", "allergic to X"). What remains
is split on commas, "and", "&" and similar. Lead-ins such as "I have" or "what can I make with" are
removed, and each chunk is normalized. Unresolved chunks are scanned for in-vocabulary n-grams and
otherwise reported back to the user.

**Diet labels.** RecipeDB has diet flags, but they were 0.0 even for clearly vegetarian dishes in the
records we inspected (e.g. Palak Paneer), so diet classes are derived from canonical ingredients with a
lexicon for meat, seafood, egg, dairy and honey. Plant phrases containing animal words (coconut milk,
peanut butter, vegetable stock, eggplant) are masked out before matching.

## D2. Hybrid retrieval and reranking

* **BM25** (rank_bm25, Okapi) runs over ingredient tokens. Each multi-word ingredient becomes one token
  plus its head noun (`red_lentil`, `lentil`).
* **Dense** retrieval uses `all-MiniLM-L6-v2` sentence embeddings of "title + cuisine + ingredient list".
  The query is "A recipe using a, b, c.". Search is exact inner product over L2-normalised vectors,
  which gives the same results as a FAISS `IndexFlatIP`. We use NumPy because importing faiss together
  with torch segfaults on macOS.
* **Reciprocal Rank Fusion**: score(d) = Σ 1/(60 + rank_s(d)) over the top 100 from each retriever.
* **Hard filters** (diet, max time, max calories, cuisine, exclusions) are applied as a mask *before*
  ranking, so every returned recipe satisfies them.
* **Coverage reranking**: score = (1 − w_cov − w_use)·RRF/max RRF + w_cov·coverage + w_use·pantry_use −
  λ·|missing|. *Coverage* is the share of the recipe's non-staple ingredients the user has. *Pantry_use*
  is the share of the user's items the recipe uses. Staples (salt, water, oil, pepper, sugar) are assumed
  to be in stock. Variety modifiers count as the same item (basmati rice ≈ rice, chicken breast ≈
  chicken), but coconut milk ≠ milk and sweet potato ≠ potato. Weights are tuned on a dev split.

**Synthetic evaluation.** Sample a recipe and hide it (the query has no title). Take 3–5 of its non-staple
ingredients as the pantry. The metrics are Recall@{1,5,10,20} and MRR of the hidden recipe, plus
user-facing quality of the top 5 (mean coverage and mean number of missing items). In the noisy
variant, each query ingredient is rewritten the way users type it: regional synonym (p = .7 where one
exists), plural, a typo (p = .15) and a quantity/prep phrase (p = .5). It is then retrieved either raw or
after D1 normalization.

## D3. Grounded generation and substitutions

* The top 3 reranked recipes go to the LLM as context, each with its ID, title, cuisine, time,
  ingredients and steps, plus what the user has, what is missing, and candidate substitutes for each
  missing item. The system prompt requires recommending only retrieved recipes, citing `[R<id>]`,
  taking steps and ingredients only from the cited recipe, and listing the missing items as given.
* **Citation check**: any `[R<id>]` not in the retrieved set is replaced with a visible "citation removed"
  marker.
* **Substitutions**:
  1. A curated table of about 90 ingredients with ratios and notes.
  2. A corpus fallback: an ingredient's substitutes are items with a similar *recipe context* (cosine of
     PPMI co-occurrence vectors) and a similar *kind* (cosine of MiniLM name embeddings) that rarely
     appear *together* with it. score = √(ctx·name) − co-occurrence rate, kept if > 0.4.

  Candidates that would break the user's diet are dropped, and items already in the pantry are
  offered first.
* **Backends**: local open LLM (Qwen2.5-1.5B-Instruct, greedy decoding), Claude through the Anthropic
  SDK, or `extractive` (a deterministic template with no LLM).
* **Plain-LLM baseline**: the same model and the same pantry text with no retrieval, asked for the same
  answer format.

## D4. Evaluation and demo

* **Faithfulness** (automatic):
  * Ingredient grounding: mentioned ingredients that are backed by the reference recipe document, the
    pantry or a suggested substitute.
  * Step grounding: generated steps whose content words are at least 50% covered by one step of the
    reference recipe.
  * Unflagged missing: ingredients used in the steps that the user lacks and the answer never listed
    as missing.
  * Diet violations.
  * Citation validity.

  For RAG the reference recipe is the cited one. For the plain LLM it is the corpus recipe with the
  highest ingredient Jaccard to the answer. This is deliberately generous to the baseline: it gets
  credit if *any* real recipe backs it up.
* **Sanity check**: the extractive generator, which copies its recipe verbatim, scores 0.98 ingredient
  grounding and 0.99 step grounding. This is the metric's practical ceiling.
* **Human evaluation**: `make_human_eval.py` builds a blinded, order-randomised A/B sheet (RAG vs plain)
  with 1–5 ratings for relevance, feasibility and correctness, plus an overall preference. It then scores
  the filled sheets.
* **Demo**: a Streamlit chat with multi-turn dialogue state. It handles adding or removing ingredients,
  changing constraints, picking the n-th recipe, substitution questions and resets, and shows an evidence
  panel with each retrieved recipe's ranks, coverage and missing items.
