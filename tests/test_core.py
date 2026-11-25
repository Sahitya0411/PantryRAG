"""Fast unit tests (no corpus or models needed)."""
from pantryrag.corpus import Recipe
from pantryrag.diet import recipe_diets
from pantryrag.generation import check_citations, extractive_answer
from pantryrag.normalize import Normalizer, canonical_form, clean, singularize
from pantryrag.pantry_parser import parse_pantry
from pantryrag.retrieval import Hit, _head_match, coverage_stats

VOCAB = ["paneer", "spinach", "rice", "onion", "tomato", "bell pepper", "yogurt", "cumin",
         "turmeric", "potato", "egg", "green onion", "soy sauce", "chickpea", "eggplant",
         "basmati rice", "heavy cream", "garlic", "chicken", "red onion", "cilantro"]
N = Normalizer(VOCAB)


def test_clean_and_singular():
    """Cleaning drops quantities/units/descriptors; singular and accent handling work."""
    assert clean("2 cups Paneer cubes, chopped") == "paneer"
    assert singularize("tomatoes") == "tomato"
    assert singularize("berries") == "berry"
    assert canonical_form("Crème Fraîche") == "creme fraiche"


def test_normalize_synonyms_and_fuzzy():
    """Regional names map via synonyms and misspellings via fuzzy matching."""
    assert N.normalize("capsicum").canonical == "bell pepper"
    assert N.normalize("curd").canonical == "yogurt"
    assert N.normalize("jeera").canonical == "cumin"
    assert N.normalize("spring onions").canonical == "green onion"
    assert N.normalize("tomatos").canonical == "tomato"
    assert N.normalize("brinjal").canonical == "eggplant"
    assert N.normalize("double cream").canonical == "heavy cream"


def test_parse_pantry_constraints():
    """Ingredients, diet, time, cuisine and exclusions are parsed from one message."""
    q = parse_pantry("I have paneer, spinach, rice, onion and tomato; vegetarian, under 30 minutes", N)
    assert q.ingredients == ["paneer", "spinach", "rice", "onion", "tomato"]
    assert q.diet == "vegetarian" and q.max_minutes == 30
    q = parse_pantry("chicken, curd and jeera - indian, no garlic, in 1 hour", N)
    assert q.cuisine == "indian" and q.exclude == ["garlic"] and q.max_minutes == 60
    assert "chicken" in q.ingredients and "garlic" not in q.ingredients


def test_diet_rules():
    """Diet labels, including plant items with animal words ("coconut milk")."""
    assert recipe_diets(["paneer", "spinach"])["vegetarian"]
    assert not recipe_diets(["paneer", "spinach"])["vegan"]
    assert recipe_diets(["coconut milk", "eggplant", "peanut butter"])["vegan"]
    assert not recipe_diets(["chicken stock", "rice"])["vegetarian"]


def test_head_match_and_coverage():
    """Variety-level matching and pantry coverage (staples like salt are ignored)."""
    assert _head_match("basmati rice", "rice")
    assert _head_match("chicken breast", "chicken")
    assert not _head_match("coconut milk", "milk")
    assert not _head_match("sweet potato", "potato")
    r = Recipe("1", "Palak Paneer", ["paneer", "spinach", "onion", "salt", "cream"], ["cook"])
    matched, missing, cov = coverage_stats(r, ["paneer", "spinach", "onion"])
    assert missing == ["cream"] and abs(cov - 0.75) < 1e-9


def test_citations_checked():
    """Citations outside the retrieved set are reported as invalid."""
    r = Recipe("42", "X", ["a", "b", "c"], ["do it"])
    hits = [Hit(r, 1.0)]
    valid, invalid = check_citations("Make X [R42], or Y [R99]", hits)
    assert valid == ["42"] and invalid == ["99"]


def test_extractive_answer_cites():
    """The no-LLM answer cites its recipe and lists the missing ingredient."""
    r = Recipe("7", "Dal", ["red lentil", "onion", "cumin"], ["boil lentils", "temper cumin"])
    q = parse_pantry("red lentil, onion", Normalizer(["red lentil", "onion", "cumin"]))
    h = Hit(r, 1.0, matched=["red lentil", "onion"], missing=["cumin"], coverage=2 / 3)
    txt = extractive_answer(q, [h], {"7": {}})
    assert "[R7]" in txt and "cumin" in txt


def test_diet_mixed_phrases():
    """A meat word next to a plant phrase is still caught ("chicken vegetable stock")."""
    assert not recipe_diets(["chicken vegetable stock"])["vegan"]
    assert recipe_diets(["vegetable stock", "butternut squash"])["vegan"]


def test_new_pantry_resets_constraints():
    """A fresh ingredient list must not inherit the previous turn's diet."""
    from types import SimpleNamespace
    from pantryrag.pipeline import ChatSession, DialogueState
    from pantryrag.pantry_parser import parse_pantry

    calls = []
    rag = SimpleNamespace(normalizer=N, parse=lambda t: parse_pantry(t, N))
    chat = ChatSession.__new__(ChatSession)
    chat.rag, chat.state = rag, DialogueState()
    chat._retrieve_and_answer = lambda t: calls.append(chat.state.query) or ("ok", None)
    chat.turn("I have paneer, spinach, rice; vegetarian")
    chat.state.query.diet = "vegan"            # as if the user said "make it vegan"
    chat.turn("chicken, capsicum, curd and jeera - indian, no cream")
    q = calls[-1]
    assert q.diet is None and "chicken" in q.ingredients and q.cuisine == "indian"
