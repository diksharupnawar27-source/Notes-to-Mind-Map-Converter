"""Streamlit UI:  streamlit run app.py"""
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import auth
import evaluate
from grader import AnswerGrader, check_blank
from llm import TextGenerator
from mindmap_view import render
from nlp_engine import NotesAnalyzer
from progress import Progress
from question_gen import QuestionGenerator
from summarizer import explain_concept, summarize_topic

st.set_page_config(page_title="Notes → Mind Map", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1250px}
.hero{padding:26px 30px;border-radius:20px;margin-bottom:18px;
 background:linear-gradient(135deg,#312e81 0%,#1e3a8a 55%,#0e7490 100%);border:1px solid #ffffff22}
.hero h1{margin:0;font-size:2.1rem;background:linear-gradient(90deg,#fff,#c4b5fd);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{margin:6px 0 0;color:#c7d2fe;font-size:1rem}
.chips{display:flex;gap:12px;margin:6px 0 16px;flex-wrap:wrap}
.chip{background:#151a2e;border:1px solid #2a3050;border-radius:14px;padding:10px 18px}
.chip b{display:block;font-size:1.5rem;color:#a5b4fc}.chip span{font-size:.78rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em}
.stTabs [data-baseweb="tab"]{font-size:1rem;padding:8px 18px}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px;border-color:#2a3050;background:#121731}
.stButton>button{border-radius:999px}
section[data-testid="stSidebar"]{border-right:1px solid #2a3050}
.qk{color:#818cf8;font-size:.75rem;text-transform:uppercase;letter-spacing:.08em}
.hero{padding:20px 30px}
.st-key-nav button{height:68px;border-radius:18px;font-size:1.02rem;font-weight:600;transition:transform .15s,box-shadow .15s}
.st-key-nav button:hover{transform:translateY(-2px);box-shadow:0 8px 22px #6366f155}
.st-key-nav button[kind="primary"]{background:linear-gradient(135deg,#6366f1,#0ea5e9);border:0;color:#fff}
.st-key-nav button[kind="secondary"]{background:#151a2e;border:1px solid #2a3050}
.feat{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:10px}
.feat div{background:#151a2e;border:1px solid #2a3050;border-radius:18px;padding:20px 22px}
.feat b{display:block;font-size:1.08rem;color:#c7d2fe;margin-bottom:6px}
.feat span{color:#94a3b8;font-size:.92rem;line-height:1.45}
</style>
<div class="hero"><h1>Notes → Mind Map</h1></div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- login / sign-up
def auth_screen():
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        t_in, t_up = st.tabs(["🔑 Log in", "📝 Sign up"])
        with t_in, st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", type="primary", width="stretch"):
                ok, val = auth.authenticate(u, p)
                if ok:
                    st.session_state.user = val
                    st.rerun()
                st.error(val)
        with t_up, st.form("signup"):
            u = st.text_input("Choose a username")
            em = st.text_input("Email (optional)")
            p = st.text_input("Password (8+ chars, letter + digit)", type="password")
            p2 = st.text_input("Confirm password", type="password")
            if st.form_submit_button("Create account", type="primary", width="stretch"):
                ok, msg = auth.register(u, em, p, p2)
                if ok:
                    st.success(msg + " Switch to the Log in tab to sign in.")
                else:
                    st.error(msg)


def logout():
    for k in list(st.session_state):
        del st.session_state[k]


if "user" not in st.session_state:
    auth_screen()
    st.stop()
user = st.session_state.user


# ---------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading transformer models (first run only)…")
def load():
    analyzer = NotesAnalyzer()
    return analyzer, QuestionGenerator(), TextGenerator(), AnswerGrader(analyzer.embedder)


analyzer, qgen, gen, grader = load()

with st.sidebar:
    st.caption(f"Signed in as **{user}**")
    st.button("Log out", on_click=logout)
    st.header("Your notes")
    title = st.text_input("Map title (optional)")
    up = st.file_uploader("Upload .txt", type="txt")
    default = up.read().decode() if up else open("sample_notes.txt", encoding="utf-8").read()
    text = st.text_area("…or paste here", default, height=330)
    go = st.button("✨ Generate", type="primary", width="stretch")

if go:
    try:
        with st.spinner("Extracting concepts, relations and questions…"):
            t0 = time.perf_counter()
            mm = analyzer.analyze(text, title or None)
            t1 = time.perf_counter()
            qs = qgen.build(mm, analyzer.embedder)
            t2 = time.perf_counter()
            st.session_state.update(
                mm=mm, text=text, html=render(mm), qs=Progress(user).sort(qs),
                results={}, summaries={}, explains={}, evaluation=None,
                timings={"mind_map_seconds": round(t1 - t0, 2), "questions_seconds": round(t2 - t1, 2)})
    except ValueError as e:
        st.error(str(e))


# ---------------------------------------------------------------- quiz helpers
def finish(rkey, concept, ok, verdict, msg):
    Progress(user).record(concept, ok)
    st.session_state.results[rkey] = {"verdict": verdict, "msg": msg}


def check_mcq(i, q):
    pick = st.session_state.get(f"mcq{i}")
    if pick is None:
        return
    ok = pick == q["mcq"]["correct"]
    finish(("mcq", i), q["key"], ok, "correct" if ok else "incorrect",
           "Correct ✅" if ok else f"Not quite — the answer is **{q['mcq']['correct']}**.")


def check_cloze(i, q):
    ok = check_blank(st.session_state.get(f"cz{i}", ""), q["term"])
    finish(("cloze", i), q["key"], ok, "correct" if ok else "incorrect",
           "Correct ✅" if ok else f"Not quite — the answer is **{q['term']}**.")


def check_written(i, q):
    r = grader.grade(st.session_state.get(f"wr{i}", ""), q["answer"], q["term"])
    finish(("wr", i), q["key"], r["verdict"] == "correct", r["verdict"],
           f"{r['feedback']}  (similarity {r['score']:.2f})")


def show_result(r):
    box = {"correct": st.success, "partial": st.warning, "incorrect": st.error}[r["verdict"]]
    _ = box(r["msg"])


def quiz(qs, mode):
    kind = {"Multiple choice": "mcq", "Fill in the blank": "cloze", "Written answer (AI-graded)": "wr"}[mode]
    usable = [(i, q) for i, q in enumerate(qs)
              if (kind == "mcq" and q["mcq"]) or (kind == "cloze" and q["cloze"]) or kind == "wr"]
    if not usable:
        st.info("No questions of this type could be built from these notes.")
        return
    res = st.session_state.results
    done = [(kind, i) in res for i, _ in usable]
    right = sum(res[(kind, i)]["verdict"] == "correct" for i, _ in usable if (kind, i) in res)
    st.progress(sum(done) / len(usable), text=f"{sum(done)} / {len(usable)} answered · {right} correct")
    for n, (i, q) in enumerate(usable, 1):
        rk, answered = (kind, i), (kind, i) in res
        with st.container(border=True):
            st.markdown(f'<div class="qk">Question {n}</div>', unsafe_allow_html=True)
            if kind == "mcq":
                st.markdown(f"**{q['mcq']['stem']}**")
                st.radio("Choose one", q["mcq"]["options"], index=None, key=f"mcq{i}", disabled=answered)
                cb = check_mcq
            elif kind == "cloze":
                st.markdown(f"**Fill in the blank:** {q['cloze']}")
                st.text_input("Your answer", key=f"cz{i}", disabled=answered)
                cb = check_cloze
            else:
                st.markdown(f"**{q['question']}**")
                st.text_area("Your answer", key=f"wr{i}", disabled=answered, height=90)
                cb = check_written
            if answered:
                show_result(res[rk])
                if kind == "wr":
                    with st.expander("Reference answer"):
                        st.write(q["answer"])
            else:
                st.button("Check answer", key=f"chk_{kind}{i}", on_click=cb, args=(i, q))


# ---------------------------------------------------------------- main view
VIEWS = [("Mind map", ":material/account_tree:"), ("AI summaries", ":material/auto_awesome:"),
         ("Revision", ":material/quiz:"), ("Evaluation", ":material/analytics:")]


def set_view(name):
    st.session_state.view = name


if "mm" in st.session_state:
    mm = st.session_state.mm
    n_con = sum(len(v) for v in mm.topics.values()) + len(mm.topics)
    st.markdown(f"""<div class="chips">
    <div class="chip"><b>{len(mm.topics)}</b><span>Topics</span></div>
    <div class="chip"><b>{n_con}</b><span>Concepts</span></div>
    <div class="chip"><b>{len(mm.relations)}</b><span>Relations</span></div>
    <div class="chip"><b>{len(st.session_state.qs)}</b><span>Questions</span></div></div>""", unsafe_allow_html=True)

    st.session_state.setdefault("view", "Mind map")
    with st.container(key="nav"):
        for col, (name, icon) in zip(st.columns(len(VIEWS)), VIEWS):
            col.button(name, icon=icon, key=f"nav_{name}", width="stretch", on_click=set_view, args=(name,),
                       type="primary" if st.session_state.view == name else "secondary")
    view = st.session_state.view
    st.write("")

    if view == "Mind map":
        components.html(st.session_state.html, height=700)

    elif view == "AI summaries":
        st.caption(f"Generator: {gen.backend} (downloaded on first use).")
        sums = st.session_state.summaries
        if st.button("✨ Summarise all topics"):
            with st.spinner("Generating summaries…"):
                for t in mm.topics:
                    sums[t] = summarize_topic(gen, mm, t)
        for t in mm.topics:
            with st.container(border=True):
                st.markdown(f"**{t}**")
                if t in sums:
                    st.write(sums[t][0])
                    st.caption(f"source: {sums[t][1]}")
                elif st.button("Summarise", key=f"sum_{t}"):
                    with st.spinner("Generating…"):
                        sums[t] = summarize_topic(gen, mm, t)
                    st.rerun()
        st.divider()
        st.subheader("Explain a concept")
        pool = {c.name: c for c in [*mm.topic_concepts.values(), *(c for k in mm.topics.values() for c in k)]}
        pick = st.selectbox("Concept", list(pool))
        if st.button("Explain"):
            with st.spinner("Generating…"):
                st.session_state.explains[pick] = explain_concept(gen, pool[pick])
        if pick in st.session_state.explains:
            txt, src = st.session_state.explains[pick]
            st.write(txt)
            st.caption(f"source: {src}")

    elif view == "Revision":
        mode = st.radio("Quiz type", ["Multiple choice", "Fill in the blank", "Written answer (AI-graded)"],
                        horizontal=True)
        quiz(st.session_state.qs, mode)

    else:
        st.caption("Compares our pipeline with a TF-IDF/K-Means baseline and checks question quality and grader calibration.")
        if st.button("▶ Run evaluation on these notes"):
            with st.spinner("Evaluating…"):
                st.session_state.evaluation = evaluate.run_evaluation(
                    st.session_state.text, analyzer, qgen, grader, mm, st.session_state.qs, st.session_state.timings)
        r = st.session_state.evaluation
        if r:
            st.markdown("**Concept extraction**")
            st.dataframe(pd.DataFrame(r["concepts"]).T, width="stretch")
            st.markdown("**Topic clustering — silhouette (higher is better)**")
            st.dataframe(pd.DataFrame([r["clusters"]]), width="stretch")
            st.markdown("**Question quality**")
            st.dataframe(pd.DataFrame([r["questions"]]).T.astype(str).rename(columns={0: "value"}), width="stretch")
            if r["grader"]:
                st.markdown("**Answer grader calibration**")
                st.dataframe(pd.DataFrame([r["grader"]]).T.astype(str).rename(columns={0: "value"}), width="stretch")
            st.markdown("**Runtime**")
            st.dataframe(pd.DataFrame([r["timings"]]), width="stretch")
            c1, c2 = st.columns(2)
            c1.download_button("⬇ Report (.md)", evaluate.to_markdown(r), "eval_report.md")
            c2.download_button("⬇ Human-rating template (.csv)",
                               pd.DataFrame(evaluate.human_eval_rows(st.session_state.qs)).to_csv(index=False),
                               "human_eval_template.csv")
else:
    st.markdown("""<div class="feat">
    <div><b>Mind map</b><span>Key concepts and their relations as an interactive, foldable map.</span></div>
    <div><b>AI summaries</b><span>Generated topic summaries and plain-language explanations.</span></div>
    <div><b>Revision quiz</b><span>Multiple choice, fill in the blank and AI-graded written answers.</span></div>
    <div><b>Evaluation</b><span>Compare the pipeline with a baseline and check question quality.</span></div>
    </div>""", unsafe_allow_html=True)
    st.write("")
    st.info("Paste or upload notes in the sidebar and press **Generate** to begin.")
