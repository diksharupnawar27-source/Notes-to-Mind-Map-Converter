"""Personalisation: Leitner-box spaced repetition, stored per user in data/progress_<user>.json."""
import json, pathlib, re, time

DIR = pathlib.Path("data")
DAYS = [0, 0, 1, 2, 4, 8]  # review gap per box


class Progress:
    def __init__(self, user: str = "guest"):
        self.path = DIR / f"progress_{re.sub(r'[^A-Za-z0-9_]', '_', user)}.json"
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}

    def record(self, key: str, correct: bool):
        e = self.d.get(key, {"box": 1})
        e["box"] = min(e["box"] + 1, 5) if correct else 1
        e["due"] = time.time() + DAYS[e["box"]] * 86400
        self.d[key] = e
        DIR.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self.d))

    def priority(self, key: str):
        e = self.d.get(key)
        if not e:
            return (1, 0)                       # new
        return (0 if e["due"] <= time.time() else 2, e["box"])  # weak/due first

    def sort(self, questions: list[dict]) -> list[dict]:
        return sorted(questions, key=lambda q: self.priority(q["key"]))
