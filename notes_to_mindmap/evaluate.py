"""Evaluation: our pipeline vs a TF-IDF/K-Means baseline, question quality and grader calibration.

CLI:  python evaluate.py [notes.txt]   -> eval_results.json, eval_report.md, human_eval_template.csv
"""
import csv
import json
import random
import sys
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from grader import PARTIAL, unit_vectors


# ---- concept extraction ---------------------------------------------------
def tfidf_concepts(sents: list[str], top_n: int = 24) -> list[str]:
    """Baseline: highest TF-IDF uni/bi-grams (no transformer, no diversity control)."""
    v = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    scores = np.asarray(v.fit_transform(sents).sum(axis=0)).ravel()
    terms = v.get_feature_names_out()
    return [terms[i] for i in np.argsort(-scores)[:top_n]]


def concept_metrics(names: list[str], sents: list[str], embedder) -> dict:
    low = [s.lower() for s in sents]
    cov = float(np.mean([any(n.lower() in s for n in names) for s in low])) if sents else 0.0
    red = 0.0
    if len(names) > 1:
        e = unit_vectors(embedder, names)
        n = len(names)
        red = float(((e @ e.T).sum() - n) / (n * (n - 1)))
    return {"concepts": len(names), "sentence_coverage": round(cov, 3),
            "redundancy (lower=better)": round(red, 3)}


def silhouette(names: list[str], labels, embedder):
    k = len(set(labels))
    if len(names) < 3 or k < 2 or k >= len(names):
        return None
    return round(float(silhouette_score(unit_vectors(embedder, names), labels, metric="cosine")), 3)


# ---- questions ------------------------------------------------------------
def question_metrics(qs: list[dict], embedder=None) -> dict:
    n = max(len(qs), 1)
    open_q = [q["question"] for q in qs]
    out = {
        "questions": len(qs),
        "well_formed (ends with ?)": round(sum(q.endswith("?") for q in open_q) / n, 3),
        "answer_grounded_in_notes": round(sum(q["term"].lower() in q["answer"].lower() for q in qs) / n, 3),
        "template_fallback_rate (lower=better)": round(sum(q.startswith("What is meant by") for q in open_q) / n, 3),
        "distinct_questions": round(len(set(open_q)) / n, 3),
        "mcq_coverage": round(sum(q["mcq"] is not None for q in qs) / n, 3),
        "cloze_coverage": round(sum(q["cloze"] is not None for q in qs) / n, 3),
    }
    mcqs = [q["mcq"] for q in qs if q["mcq"]]
    if mcqs and embedder is not None:
        sims = []
        for m in mcqs:
            e = unit_vectors(embedder, m["options"])
            c = m["options"].index(m["correct"])
            sims += [float(e[c] @ e[j]) for j in range(len(m["options"])) if j != c]
        out["distractor_similarity_to_answer"] = round(float(np.mean(sims)), 3)
        out["duplicate_options"] = sum(len(set(m["options"])) < len(m["options"]) for m in mcqs)
    return out


def human_eval_rows(qs: list[dict]) -> list[dict]:
    """Template for manual rating (1-5) - fill in and average for the report."""
    return [{"question": q["question"], "reference_answer": q["answer"],
             "clarity_1to5": "", "relevance_1to5": "", "answerable_from_notes_1to5": ""} for q in qs]


# ---- grader calibration ---------------------------------------------------
def grader_metrics(grader, mm, seed: int = 0) -> dict | None:
    """Synthetic check: shortened copies of the reference should outscore other concepts' answers."""
    rng = random.Random(seed)
    concepts = list(mm.topic_concepts.values()) + [c for k in mm.topics.values() for c in k]
    items = [(c.name, c.sentences[0]) for c in concepts]
    pos, neg = [], []
    for name, ref in items:
        others = [s for _, s in items if s != ref]
        if not others:
            continue
        kept = [w for w in ref.split() if rng.random() < 0.65]
        student = " ".join(kept) if len(kept) >= 3 else ref
        pos.append(grader.score(student, ref, name))
        neg.append(grader.score(rng.choice(others), ref, name))
    if not pos:
        return None
    auc = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    acc = (sum(p >= PARTIAL for p in pos) + sum(q < PARTIAL for q in neg)) / (len(pos) + len(neg))
    return {"pairs": len(pos), "mean_score_good_answers": round(float(np.mean(pos)), 3),
            "mean_score_wrong_answers": round(float(np.mean(neg)), 3),
            "auc": round(float(auc), 3), f"accuracy@{PARTIAL}": round(float(acc), 3),
            "note": "Synthetic proxy (shortened copies vs other concepts' sentences); "
                    "validate with real student answers for a stronger claim."}


# ---- orchestration --------------------------------------------------------
def run_evaluation(text, analyzer, qgen, grader, mm=None, qs=None, timings=None) -> dict:
    if mm is None or timings is None:
        t = time.perf_counter()
        mm = analyzer.analyze(text)
        t_map = time.perf_counter() - t
        t = time.perf_counter()
        qs = qgen.build(mm, analyzer.embedder)
        timings = {"mind_map_seconds": round(t_map, 2), "questions_seconds": round(time.perf_counter() - t, 2)}
    sents = analyzer.sentences(text)
    emb = analyzer.embedder
    ours = list(mm.topic_concepts) + [c.name for k in mm.topics.values() for c in k]
    t = time.perf_counter()
    base = tfidf_concepts(sents, len(ours))
    base_s = time.perf_counter() - t

    groups = [[t_, *(c.name for c in kids)] for t_, kids in mm.topics.items()]
    flat = [n for g in groups for n in g]
    labels = [i for i, g in enumerate(groups) for _ in g]
    k = len(groups)
    base_labels = KMeans(n_clusters=min(k, max(len(base) - 1, 1)), n_init=10, random_state=0).fit_predict(
        TfidfVectorizer().fit_transform(base)) if len(base) > 2 else [0] * len(base)
    return {
        "meta": {"sentences": len(sents), "topics": k, "relations": len(mm.relations)},
        "concepts": {"ours (KeyBERT+MMR)": concept_metrics(ours, sents, emb),
                     "baseline (TF-IDF)": concept_metrics(base, sents, emb)},
        "clusters": {"ours (MiniLM+agglomerative)": silhouette(flat, labels, emb),
                     "baseline (TF-IDF+KMeans)": silhouette(base, base_labels, emb)},
        "questions": question_metrics(qs, emb),
        "grader": grader_metrics(grader, mm),
        "timings": {**timings, "baseline_concepts_seconds": round(base_s, 3)},
    }


def to_markdown(r: dict) -> str:
    def table(d):
        cols = list(next(iter(d.values())).keys())
        rows = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        rows += [f"| {k} | " + " | ".join(str(v[c]) for c in cols) + " |" for k, v in d.items()]
        return "\n".join(rows)

    def kv(d):
        return "\n".join(f"- **{k}**: {v}" for k, v in d.items())

    g = r["grader"]
    return "\n\n".join([
        "# Evaluation report",
        f"Notes: {r['meta']['sentences']} sentences, {r['meta']['topics']} topics, {r['meta']['relations']} relations.",
        "## Concept extraction (higher coverage, lower redundancy is better)", table(r["concepts"]),
        "## Topic clustering - silhouette in embedding space (higher is better)", kv(r["clusters"]),
        "## Question quality (automatic checks)", kv(r["questions"]),
        "## Answer grader calibration", kv(g) if g else "Not enough concepts.",
        "## Runtime", kv(r["timings"]),
    ])


def main():
    from grader import AnswerGrader
    from nlp_engine import NotesAnalyzer
    from question_gen import QuestionGenerator

    path = sys.argv[1] if len(sys.argv) > 1 else "sample_notes.txt"
    text = open(path, encoding="utf-8").read()
    analyzer, qgen = NotesAnalyzer(), QuestionGenerator()
    grader = AnswerGrader(analyzer.embedder)
    t = time.perf_counter()
    mm = analyzer.analyze(text)
    t_map = time.perf_counter() - t
    t = time.perf_counter()
    qs = qgen.build(mm, analyzer.embedder)
    timings = {"mind_map_seconds": round(t_map, 2), "questions_seconds": round(time.perf_counter() - t, 2)}
    res = run_evaluation(text, analyzer, qgen, grader, mm, qs, timings)
    json.dump(res, open("eval_results.json", "w"), indent=2)
    open("eval_report.md", "w", encoding="utf-8").write(to_markdown(res))
    rows = human_eval_rows(qs)
    with open("human_eval_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(to_markdown(res))


if __name__ == "__main__":
    main()
