# Notes → Mind Map Converter
NLP + transformer GenAI app: lecture notes -> concepts -> interactive mind map -> AI summaries -> adaptive quiz with answer checking.

## Features
| Feature | Tech |
|---|---|
| Sign up / log in (per-user progress) | SQLite + salted PBKDF2-SHA256 (`auth.py`, `users.db`) |
| Sentence split, relations (subject-verb-object) | spaCy `en_core_web_sm` |
| Key concepts | KeyBERT + `all-MiniLM-L6-v2` (MMR for diversity) |
| Hierarchy | Sentence embeddings + agglomerative clustering; cluster centroid = topic |
| Interactive mind map | D3 (loaded from a CDN, needs internet) |
| **AI summaries / "explain this concept"** | Local Flan-T5 (`google/flan-t5-base`) with an extractive fallback (`llm.py`, `summarizer.py`) |
| Open question generation | T5 `valhalla/t5-small-qg-hl` |
| **Multiple-choice quiz** | Correct term + 3 semantically closest other concepts as distractors |
| **Fill-in-the-blank quiz** | Concept blanked out of its source sentence, fuzzy-matched |
| **Answer checking** | Written answers graded by embedding similarity to the notes (`grader.py`) |
| Personalisation | Leitner spaced repetition, one file per user in `data/` |
| **Evaluation** | `evaluate.py` + in-app tab: baseline comparison, question quality, grader calibration |

## Run
```
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```
First run downloads the models from huggingface.co (~300 MB; ~1 GB more the first time the local summariser is used).
Create an account on the **Sign up** tab, then log in.

### Summaries
Summaries use a local model, `google/flan-t5-base` by default (set the `LOCAL_LLM` environment variable to use another
Flan-T5 variant). If it cannot be loaded, the app falls back to plain extractive summaries, labelled as such.

## Evaluation
In the app: **Evaluation** tab -> *Run evaluation*. From the command line:
```
python evaluate.py sample_notes.txt
```
writes `eval_results.json`, `eval_report.md` and `human_eval_template.csv` (rate each question 1-5, then average for the report).
It compares our concept extraction (KeyBERT+MMR) and clustering (MiniLM+agglomerative) with a TF-IDF / K-Means baseline,
checks question quality automatically, and calibrates the answer grader. The grader check is a synthetic proxy; for a stronger
claim, collect real student answers and rate them manually.

## Notes
- Accounts are local (`users.db`); there is no password reset or persistent "remember me" - a browser refresh logs you out.
- Delete `users.db` and `data/` to reset all accounts and progress.
