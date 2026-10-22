"""Answer checking: semantic grading for written answers, fuzzy matching for fill-in-the-blank."""
import difflib
import re

import numpy as np

CORRECT, PARTIAL = 0.60, 0.40       # similarity thresholds (calibrated in evaluate.py)


def unit_vectors(embedder, texts) -> np.ndarray:
    e = np.asarray(embedder.encode(list(texts)), dtype=float)
    return e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-9, None)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())     # ignore case, spaces, hyphens, punctuation


def check_blank(student: str, answer: str, cutoff: float = 0.9) -> bool:
    a, b = _norm(student), _norm(answer)
    return bool(a) and (a == b or difflib.SequenceMatcher(None, a, b).ratio() >= cutoff)


class AnswerGrader:
    def __init__(self, embedder):
        self.embedder = embedder

    def score(self, student: str, reference: str, concept: str | None = None) -> float:
        """Cosine similarity of the two answers (+0.1 if the key term is mentioned), in [0, 1]."""
        student = student.strip()
        if not student:
            return 0.0
        e = unit_vectors(self.embedder, [student, reference])
        s = float(e[0] @ e[1])
        if concept and concept.lower() in student.lower():
            s += 0.10
        return max(0.0, min(1.0, s))

    def grade(self, student: str, reference: str, concept: str | None = None) -> dict:
        s = self.score(student, reference, concept)
        verdict = "correct" if s >= CORRECT else "partial" if s >= PARTIAL else "incorrect"
        feedback = {"correct": "Great answer - it matches the notes.",
                    "partial": "Partly there - compare with the reference answer below.",
                    "incorrect": "Not quite - read the reference answer below and try again later."}[verdict]
        return {"score": round(s, 2), "verdict": verdict, "feedback": feedback}
