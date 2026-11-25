"""Polite, resumable scraper for BBC Good Food recipe pages.

Reads recipe URLs from the group's existing CSV (recipes_raw.csv), fetches each
page and keeps only the schema.org Recipe JSON-LD block: ingredients, steps,
prep/cook/total time, nutrition, yield, diet tags, keywords, category, cuisine.

Respects robots.txt: /recipes/ is allowed, and the site asks for a 12-second
crawl delay, so we wait 12 s between requests (≈300 pages/hour). The site's
content signal is `search=yes, ai-train=no`: we build a retrieval index and
never train a model on the text.

    python scripts/scrape_bbc.py --csv /path/to/recipes_raw.csv --max 600
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "bbc_scraped.jsonl"
UA = "PantryRAG-academic-crawler/1.0 (course project; polite, 12s delay)"
LD_RE = re.compile(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', re.S)
KEEP = ["name", "recipeIngredient", "recipeInstructions", "prepTime", "cookTime", "totalTime",
        "nutrition", "recipeYield", "suitableForDiet", "keywords", "recipeCategory",
        "recipeCuisine", "description"]


def extract_recipe(html: str) -> dict | None:
    for m in LD_RE.finditer(html):
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        items = d if isinstance(d, list) else d.get("@graph", [d])
        for it in items:
            if isinstance(it, dict) and "Recipe" in str(it.get("@type")):
                rec = {k: it.get(k) for k in KEEP if it.get(k) is not None}
                steps = []
                for s in rec.get("recipeInstructions") or []:
                    t = s.get("text") if isinstance(s, dict) else s
                    if t:
                        steps.append(re.sub(r"<[^>]+>", "", str(t)).strip())
                rec["recipeInstructions"] = steps
                return rec
    return None


def fetch(url: str, timeout: int = 30) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return 0, ""


def ordered_urls(csv: Path, seed: int = 43) -> list[str]:
    df = pd.read_csv(csv)
    df = df[df.recipe_url.str.contains("/recipes/", na=False)]
    cuisine = df.cuisine.fillna("").str.lower()
    pri = [df[cuisine.str.contains("indian")],                         # demo-critical first
           df[~cuisine.str.contains("indian") & (cuisine != "")],     # other tagged cuisines
           df[cuisine == ""]]
    urls = []
    rng = random.Random(seed)
    for part in pri:
        u = part.recipe_url.tolist()
        rng.shuffle(u)
        urls += u
    return urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max", type=int, default=600, help="pages to fetch this run")
    ap.add_argument("--delay", type=float, default=12.0, help="robots.txt Crawl-delay")
    args = ap.parse_args()
    if args.delay < 12:
        sys.exit("refusing: robots.txt asks for a 12 s crawl delay")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["url"] for l in OUT.read_text().splitlines() if l.strip()}
    todo = [u for u in ordered_urls(Path(args.csv)) if u not in done][: args.max]
    print(f"{len(done)} already scraped, fetching {len(todo)} (~{len(todo)*args.delay/3600:.1f} h)", flush=True)
    fails = 0
    with open(OUT, "a") as f:
        for i, url in enumerate(todo):
            t0 = time.time()
            status, html = fetch(url)
            rec = extract_recipe(html) if status == 200 else None
            if rec:
                f.write(json.dumps({"url": url, **rec}) + "\n")
                f.flush()
            else:
                fails += 1
                if status in (403, 429):  # being told to back off: stop politely
                    print(f"HTTP {status} at {url}; stopping", flush=True)
                    break
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} fetched, {fails} failed", flush=True)
            time.sleep(max(0.0, args.delay - (time.time() - t0)))
    print(f"done; failures={fails}", flush=True)


if __name__ == "__main__":
    main()
