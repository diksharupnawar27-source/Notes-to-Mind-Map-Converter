"""Question creation: T5 for open questions, plus multiple-choice and fill-in-the-blank
variants built from the same concepts (distractors = semantically closest other concepts)."""
import random
import re

import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from grader import unit_vectors


class QuestionGenerator:
    def __init__(self, model: str = "valhalla/t5-small-qg-hl"):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)

    def ask(self, context: str, answer: str) -> str:
        marked = re.sub(re.escape(answer), f"<hl> {answer} <hl>", context, count=1, flags=re.I)
        ids = self.tok("generate question: " + marked, return_tensors="pt",
                       truncation=True, max_length=256)
        out = self.model.generate(**ids, max_length=48, num_beams=2)
        q = self.tok.decode(out[0], skip_special_tokens=True).strip()
        return q if q.endswith("?") else f"What is meant by '{answer}' in this context?"

    @staticmethod
    def cloze(context: str, answer: str) -> str | None:
        out, n = re.subn(re.escape(answer), "_____", context, count=1, flags=re.I)
        return out if n else None

    @staticmethod
    def distractors(i: int, names: list[str], emb: np.ndarray, k: int = 3) -> list[str]:
        me = names[i].lower()
        out = []
        for j in np.argsort(-(emb @ emb[i])):
            n = names[j].lower()
            if j == i or n in me or me in n:          # skip itself and near-duplicates
                continue
            out.append(names[j])
            if len(out) == k:
                break
        return out

    def build(self, mindmap, embedder=None, limit: int = 12) -> list[dict]:
        pool = list(mindmap.topic_concepts.values())
        for kids in mindmap.topics.values():
            pool += kids
        pool.sort(key=lambda c: -c.score)
        names = [c.name for c in pool]
        emb = unit_vectors(embedder, names) if embedder is not None and len(pool) > 3 else None
        qs = []
        for i, c in enumerate(pool[:limit]):
            ctx = c.sentences[0]
            q = self.ask(ctx, c.name)
            blank = self.cloze(ctx, c.name)
            item = {"key": c.name, "term": c.name, "question": q, "answer": ctx,
                    "cloze": blank, "mcq": None}
            leaks = c.name.lower() in q.lower()           # question would reveal the answer
            stem = q if not leaks else (f"Which term completes this statement?\n\n{blank}" if blank else None)
            if emb is not None and stem:
                wrong = self.distractors(i, names, emb)
                if len(wrong) == 3:
                    options = [c.name, *wrong]
                    random.Random(c.name).shuffle(options)   # stable across Streamlit reruns
                    item["mcq"] = {"stem": stem, "options": options, "correct": c.name}
            qs.append(item)
        return qs
