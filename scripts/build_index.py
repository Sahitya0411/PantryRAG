"""Build processed corpus + BM25/dense index.

    python scripts/build_index.py --source recipedb   # or --source bbc (stand-in)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pantryrag import corpus  # noqa: E402
from pantryrag.retrieval import Retriever  # noqa: E402

# 1) clean the raw corpus and save it, 2) embed every recipe and save the dense index
ap = argparse.ArgumentParser()
ap.add_argument("--source", default="recipedb", choices=list(corpus.RAW_SOURCES))
args = ap.parse_args()
recipes, freq = corpus.build(args.source)
corpus.save(recipes, freq)
# record which corpus was built so evaluations and the UI can report it
(corpus.PROCESSED / "SOURCE").write_text(args.source)
print(f"{len(recipes)} recipes, {len(freq)} ingredient types")
Retriever(recipes).save()
print("index saved to", corpus.PROCESSED / "index")
