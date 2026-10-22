"""Text-generation backend for summaries and explanations: a local Flan-T5 model.

The model (LOCAL_LLM env var, default google/flan-t5-base) is downloaded and loaded
lazily, on the first summary or explanation request.
"""
import os


class TextGenerator:
    def __init__(self):
        self.local_name = os.getenv("LOCAL_LLM", "google/flan-t5-base")
        self._tok = self._model = None

    @property
    def backend(self) -> str:
        return f"local {self.local_name}"

    def generate(self, prompt: str, max_tokens: int = 160) -> str:
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.local_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.local_name)
        ids = self._tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        out = self._model.generate(**ids, max_new_tokens=max_tokens, num_beams=2,
                                   no_repeat_ngram_size=3)
        return self._tok.decode(out[0], skip_special_tokens=True).strip()
