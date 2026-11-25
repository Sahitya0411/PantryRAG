"""D4: build a blinded human-evaluation sheet from cached generations, and
score it once raters have filled it in.

    python scripts/make_human_eval.py build --backend hf --n 20
        -> results/human_eval/sheet.csv  (give to raters; A/B order randomised)
           results/human_eval/key.json   (which of A/B is RAG; keep private)
    python scripts/make_human_eval.py score results/human_eval/sheet_rater1.csv [...]
        -> mean ratings per system + RAG win rate, printed and saved

Each rater scores both answers 1-5 on: relevance (uses my pantry), feasibility
(I could actually cook this), correctness (ingredients/steps make sense,
nothing invented), and picks an overall preference (A/B/tie).
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/human_eval"
CRIT = ["relevance", "feasibility", "correctness"]


def build(backend: str, n: int, seed: int = 43):
    """Write a blinded rating sheet for `n` queries; RAG/plain are randomly assigned
    to columns A/B and the assignment is saved separately in key.json."""
    recs = [json.loads(l) for l in open(ROOT / f"results/generations_{backend}.jsonl")]
    by_q = defaultdict(dict)
    for r in recs:
        by_q[r["query"]][r["system"]] = r["text"]
    qs = [q for q, d in by_q.items() if "rag" in d and "plain" in d]
    rng = random.Random(seed)
    qs = rng.sample(qs, min(n, len(qs)))
    OUT.mkdir(parents=True, exist_ok=True)
    key = {}
    with open(OUT / "sheet.csv", "w", newline="") as f:
        cols = ["item", "pantry_query", "answer_A", "answer_B"]
        cols += [f"{c}_{s}" for s in "AB" for c in CRIT] + ["preference(A/B/tie)", "comments"]
        w = csv.writer(f)
        w.writerow(cols)
        for i, q in enumerate(qs, 1):
            rag_first = rng.random() < 0.5
            a, b = (by_q[q]["rag"], by_q[q]["plain"]) if rag_first else (by_q[q]["plain"], by_q[q]["rag"])
            key[str(i)] = "A" if rag_first else "B"
            w.writerow([i, q, a, b] + [""] * (len(CRIT) * 2 + 2))
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    print(f"wrote {len(qs)} items to {OUT/'sheet.csv'} (key in key.json)")


def score(paths: list[str]):
    """Un-blind the filled sheets with key.json and report mean ratings per system
    and the share of preferences for RAG, plain and tie."""
    key = json.loads((OUT / "key.json").read_text())
    vals = {s: defaultdict(list) for s in ("rag", "plain")}
    pref = defaultdict(int)
    for p in paths:
        for row in csv.DictReader(open(p)):
            rag_slot = key[str(row["item"])]
            plain_slot = "B" if rag_slot == "A" else "A"
            for c in CRIT:
                for sys_, slot in (("rag", rag_slot), ("plain", plain_slot)):
                    v = row.get(f"{c}_{slot}", "").strip()
                    if v:
                        vals[sys_][c].append(float(v))
            pr = row.get("preference(A/B/tie)", "").strip().upper()
            if pr:
                pref["rag" if pr == rag_slot else "tie" if pr.startswith("T") else "plain"] += 1
    res = {s: {c: round(sum(v) / len(v), 2) for c, v in d.items() if v} for s, d in vals.items()}
    tot = sum(pref.values()) or 1
    res["preference"] = {k: round(v / tot, 3) for k, v in pref.items()}
    res["n_ratings"] = tot
    print(json.dumps(res, indent=2))
    (OUT / "scores.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--backend", default="hf")
    b.add_argument("--n", type=int, default=20)
    s = sub.add_parser("score")
    s.add_argument("sheets", nargs="+")
    a = ap.parse_args()
    build(a.backend, a.n) if a.cmd == "build" else score(a.sheets)
    sys.exit(0)
