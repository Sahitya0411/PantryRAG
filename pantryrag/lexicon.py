"""Hand-written lexical resources: synonyms, descriptors, units, diet classes, staples."""

# Surface form -> canonical RecipeDB-style ingredient name.
# Covers Indian/British/US regional names and common shorthands.
SYNONYMS: dict[str, str] = {
    # Indian names
    "aloo": "potato", "alu": "potato", "batata": "potato",
    "pyaz": "onion", "pyaaz": "onion", "kanda": "onion",
    "tamatar": "tomato", "lehsun": "garlic", "lasun": "garlic", "adrak": "ginger",
    "palak": "spinach", "methi": "fenugreek leaf", "kasuri methi": "fenugreek leaf",
    "methi seeds": "fenugreek seed", "gobi": "cauliflower", "phool gobi": "cauliflower",
    "patta gobi": "cabbage", "bandh gobi": "cabbage", "bhindi": "okra", "lady finger": "okra",
    "ladies finger": "okra", "baingan": "eggplant", "brinjal": "eggplant", "aubergine": "eggplant",
    "shimla mirch": "bell pepper", "capsicum": "bell pepper", "mirch": "green chile pepper",
    "hari mirch": "green chile pepper", "green chilli": "green chile pepper",
    "green chili": "green chile pepper", "green chillies": "green chile pepper",
    "chilli": "chile pepper", "chili": "chile pepper", "chillies": "chile pepper",
    "lal mirch": "red chile powder", "red chilli powder": "red chile powder",
    "chilli powder": "chili powder", "mattar": "pea", "matar": "pea", "peas": "pea",
    "gajar": "carrot", "mooli": "radish", "kheera": "cucumber", "lauki": "bottle gourd",
    "karela": "bitter melon", "bitter gourd": "bitter melon", "nimbu": "lemon",
    "dhania": "cilantro", "hara dhania": "cilantro", "coriander leaves": "cilantro",
    "coriander leaf": "cilantro", "fresh coriander": "cilantro", "dhania powder": "ground coriander",
    "coriander powder": "ground coriander", "jeera": "cumin", "zeera": "cumin",
    "cumin seeds": "cumin", "jeera powder": "ground cumin", "haldi": "turmeric",
    "rai": "mustard seed", "sarson": "mustard seed", "elaichi": "cardamom",
    "laung": "clove", "dalchini": "cinnamon", "tej patta": "bay leaf", "hing": "asafoetida",
    "ajwain": "carom seed", "saunf": "fennel seed", "kalonji": "nigella seed",
    "imli": "tamarind", "gur": "jaggery", "dahi": "yogurt", "curd": "yogurt",
    "doodh": "milk", "malai": "cream", "makhan": "butter", "ghee": "ghee",
    "paneer": "paneer", "cottage cheese": "paneer", "chawal": "rice", "basmati": "basmati rice",
    "atta": "whole wheat flour", "maida": "all purpose flour", "plain flour": "all purpose flour",
    "besan": "chickpea flour", "gram flour": "chickpea flour", "sooji": "semolina",
    "suji": "semolina", "rava": "semolina", "poha": "flattened rice",
    "chana": "chickpea", "kabuli chana": "chickpea", "chole": "chickpea",
    "garbanzo": "chickpea", "garbanzo bean": "chickpea", "rajma": "kidney bean",
    "moong dal": "mung bean", "moong": "mung bean", "masoor dal": "red lentil",
    "masoor": "red lentil", "toor dal": "pigeon pea", "arhar dal": "pigeon pea",
    "urad dal": "black gram", "dal": "lentil", "daal": "lentil",
    "murgi": "chicken", "murgh": "chicken", "gosht": "lamb", "mutton": "lamb",
    "anda": "egg", "ande": "egg", "machli": "fish", "jhinga": "shrimp",
    "nariyal": "coconut", "kaju": "cashew", "badam": "almond", "kishmish": "raisin",
    "pudina": "mint", "kadi patta": "curry leaf", "curry leaves": "curry leaf",
    "kari patta": "curry leaf", "garam masala powder": "garam masala",
    # British/US variants
    "courgette": "zucchini", "spring onion": "green onion", "scallion": "green onion",
    "scallions": "green onion", "spring onions": "green onion", "rocket": "arugula",
    "prawn": "shrimp", "prawns": "shrimp", "mince": "ground beef", "minced beef": "ground beef",
    "beef mince": "ground beef", "minced meat": "ground beef", "keema": "ground lamb",
    "coriander": "cilantro", "double cream": "heavy cream", "single cream": "light cream",
    "caster sugar": "white sugar", "icing sugar": "confectioners sugar",
    "powdered sugar": "confectioners sugar", "sugar": "white sugar", "bicarbonate of soda": "baking soda",
    "bicarb": "baking soda", "cornflour": "cornstarch", "corn flour": "cornstarch",
    "beetroot": "beet", "swede": "rutabaga", "mangetout": "snow pea",
    "chickpeas": "chickpea", "garbanzos": "chickpea", "tinned tomato": "canned tomato",
    "tomato puree": "tomato paste", "passata": "tomato sauce", "maize": "corn",
    "sweetcorn": "corn", "sweet corn": "corn", "bell peppers": "bell pepper",
    "red capsicum": "red bell pepper", "green capsicum": "green bell pepper",
    "yoghurt": "yogurt", "greek yoghurt": "greek yogurt", "eggs": "egg",
    "oil": "vegetable oil", "cooking oil": "vegetable oil", "refined oil": "vegetable oil",
    "evoo": "olive oil", "extra virgin olive oil": "olive oil", "soy": "soy sauce",
    "soya sauce": "soy sauce", "tofu": "tofu", "bean curd": "tofu",
    "noodles": "noodle", "pasta": "pasta", "spaghetti": "spaghetti",
    "bread crumbs": "bread crumb", "breadcrumbs": "bread crumb",
    "chicken breast": "chicken breast", "boneless chicken": "chicken",
    "salt and pepper": "salt", "pepper": "black pepper", "black pepper powder": "black pepper",
    "kali mirch": "black pepper", "jaggery": "jaggery", "brown onion": "onion",
    "red onion": "red onion", "white onion": "onion", "shallots": "shallot",
    "garlic clove": "garlic", "garlic cloves": "garlic", "tomato pur": "tomato paste",
    "tomato puree_": "tomato paste", "confectioner sugar": "confectioners sugar",
}

# Words that describe preparation/state, not identity.
DESCRIPTORS = {
    "fresh", "freshly", "chopped", "finely", "roughly", "coarsely", "diced", "sliced", "thinly",
    "minced", "grated", "shredded", "crushed", "ground_", "peeled", "deveined", "boneless",
    "skinless", "cubed", "cubes", "cube", "pieces", "piece", "large", "small", "medium", "big",
    "ripe", "frozen", "thawed", "cooked", "uncooked", "raw", "leftover", "dried_", "packed",
    "softened", "melted", "beaten", "whole_", "halved", "quartered", "trimmed", "washed",
    "rinsed", "drained", "organic", "homemade", "store", "bought", "some", "few", "a", "an",
    "the", "of", "bunch", "handful", "pinch", "dash", "can", "cans", "tin", "tins", "packet",
    "pack", "bag", "box", "jar", "bottle", "leftovers", "approx", "about", "around", "or",
    "optional", "to", "taste", "my", "fridge", "pantry", "plus", "extra", "good", "quality",
    "mashed", "boiled", "roasted", "toasted", "julienned", "sprig", "sprigs", "clove_",
    "stick", "sticks", "head", "heads", "stalk", "stalks", "fillet", "fillets", "breast_",
    "baby", "bit", "bits", "leftover", "half", "quarter", "lots", "lot", "little",
}

UNITS = {
    "g", "gm", "gms", "gram", "grams", "kg", "kgs", "kilo", "ml", "l", "litre", "liter",
    "litres", "liters", "cup", "cups", "tbsp", "tablespoon", "tablespoons", "tsp",
    "teaspoon", "teaspoons", "oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds",
    "pcs", "pc", "nos", "dozen", "x",
}

# Diet lexicon applied to canonical ingredient names (substring match on whole words).
MEAT = {"chicken", "beef", "pork", "lamb", "mutton", "goat", "veal", "turkey", "duck",
        "bacon", "ham", "sausage", "pepperoni", "salami", "prosciutto", "chorizo", "venison",
        "steak", "meat", "rabbit", "quail", "goose", "brisket", "rib", "ribs", "hot dog",
        "meatball", "gelatin", "lard", "pancetta", "bratwurst", "kielbasa", "liver",
        "oxtail", "mince", "keema", "veal", "bone broth", "beef broth", "chicken broth",
        "chicken stock", "beef stock", "bouillon", "drumstick", "wing", "thigh"}
SEAFOOD = {"fish", "salmon", "tuna", "cod", "shrimp", "prawn", "crab", "lobster", "anchovy",
           "sardine", "tilapia", "mackerel", "clam", "mussel", "oyster", "scallop", "squid",
           "octopus", "halibut", "trout", "catfish", "snapper", "fish sauce", "seafood",
           "haddock", "pollock", "herring", "caviar", "roe", "shellfish", "bass", "swordfish",
           "mahi", "crawfish", "calamari", "surimi", "bonito", "dashi"}
EGG = {"egg", "eggs", "mayonnaise", "meringue"}
DAIRY = {"milk", "cheese", "butter", "cream", "yogurt", "ghee", "paneer", "curd", "whey",
         "buttermilk", "mozzarella", "parmesan", "cheddar", "ricotta", "feta", "khoa",
         "condensed milk", "half-and-half", "half and half", "casein", "custard", "ice cream"}
OTHER_ANIMAL = {"honey"}
# Plant items whose names contain an animal word.
NOT_ANIMAL = {"coconut milk", "almond milk", "soy milk", "oat milk", "rice milk",
              "cashew milk", "peanut butter", "almond butter", "cocoa butter", "apple butter",
              "butternut squash", "butternut", "butter bean", "cream of tartar", "eggplant",
              "egg noodle_", "vegan", "vegetable broth", "vegetable stock", "mushroom",
              "coconut cream", "cream soda", "shea butter", "nut butter", "sunflower butter",
              "butterhead lettuce", "chicken pea", "buttercup squash", "creamed corn_"}

# Assumed always available; excluded from "missing ingredient" lists.
STAPLES = {"salt", "water", "black pepper", "vegetable oil", "oil", "ice", "white sugar",
           "cooking spray", "ice cube", "pepper", "sugar", "olive oil", "canola oil",
           "kosher salt", "sea salt", "salt and pepper"}
