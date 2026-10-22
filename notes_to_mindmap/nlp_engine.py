"""NLP + transformer pipeline: notes -> concepts -> hierarchy -> relations."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


@dataclass
class Concept:
    name: str
    sentences: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class MindMap:
    root: str
    topics: dict[str, list[Concept]]            # topic label -> child concepts
    topic_concepts: dict[str, Concept]          # topic label -> its own Concept
    relations: list[tuple[str, str, str]]       # (subject, verb, object)


class NotesAnalyzer:
    def __init__(self, embed_model: str = "all-MiniLM-L6-v2"):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise SystemExit("Run: python -m spacy download en_core_web_sm")
        self.embedder = SentenceTransformer(embed_model)   # transformer encoder
        self.kb = KeyBERT(model=self.embedder)             # BERT-based keyphrases

    # ---- 1. preprocessing -------------------------------------------------
    def sentences(self, text: str) -> list[str]:
        text = re.sub(r"^\s*([-*•]|\d+[.)]|#+)\s*", "", text, flags=re.M)
        doc = self.nlp(text)
        return [s.text.strip() for s in doc.sents if len(s.text.split()) >= 4]

    # ---- 2. key concept extraction ---------------------------------------
    def concepts(self, text: str, sents: list[str], top_n: int = 24) -> list[Concept]:
        kws = self.kb.extract_keywords(
            text, keyphrase_ngram_range=(1, 3), stop_words="english",
            use_mmr=True, diversity=0.5, top_n=top_n)
        out = []
        for kw, score in kws:
            hits = [s for s in sents if kw.lower() in s.lower()]
            if hits:
                out.append(Concept(kw, hits, float(score)))
        return out

    # ---- 3. hierarchy via semantic clustering ----------------------------
    def hierarchy(self, concepts: list[Concept], title: str | None = None):
        n = len(concepts)
        k = max(2, min(n, round(n ** 0.5)))
        emb = self.embedder.encode([c.name for c in concepts])
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(emb)
        topics, topic_concepts = {}, {}
        for cl in range(k):
            idx = [i for i, l in enumerate(labels) if l == cl]
            if not idx:
                continue
            centre = emb[idx].mean(axis=0)
            head = min(idx, key=lambda i: ((emb[i] - centre) ** 2).sum())
            topic_concepts[concepts[head].name] = concepts[head]
            topics[concepts[head].name] = [concepts[i] for i in idx if i != head]
        root = title or self.kb.extract_keywords(
            " ".join(c.name for c in concepts), keyphrase_ngram_range=(1, 2),
            stop_words="english", top_n=1)[0][0].title()
        return root, topics, topic_concepts

    # ---- 4. relation extraction (dependency parse SVO) -------------------
    def relations(self, sents: list[str], names: list[str]):
        low = {n.lower(): n for n in names}

        def match(span: str):
            for l, n in low.items():
                if l in span.lower():
                    return n

        rels = []
        for s in sents:
            for tok in self.nlp(s):
                if tok.pos_ not in ("VERB", "AUX"):
                    continue
                subj = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
                obj = [c for c in tok.children if c.dep_ in ("dobj", "attr", "acomp")]
                obj += [g for c in tok.children if c.dep_ == "prep" for g in c.children if g.dep_ == "pobj"]
                for a in subj:
                    for b in obj:
                        x = match(" ".join(t.text for t in a.subtree))
                        y = match(" ".join(t.text for t in b.subtree))
                        if x and y and x != y:
                            rels.append((x, tok.lemma_, y))
        return list(dict.fromkeys(rels))

    # ---- full pipeline ----------------------------------------------------
    def analyze(self, text: str, title: str | None = None) -> MindMap:
        sents = self.sentences(text)
        cons = self.concepts(text, sents)
        if len(cons) < 3:
            raise ValueError("Need a few more lines of notes to find concepts.")
        root, topics, tc = self.hierarchy(cons, title)
        names = [c.name for c in cons]
        return MindMap(root, topics, tc, self.relations(sents, names))
