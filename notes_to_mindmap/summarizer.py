"""Generative summaries per topic and 'explain this concept' answers, grounded in the notes."""


def _context(concepts, limit: int = 1400) -> str:
    seen, sents = set(), []
    for c in concepts:
        for s in c.sentences:
            if s not in seen:
                seen.add(s)
                sents.append(s)
    return " ".join(sents)[:limit]


def _extractive(context: str, n: int = 2) -> str:
    return " ".join(context.replace("\n", " ").split(". ")[:n]).strip()


def _run(gen, prompt: str, context: str, max_tokens: int) -> tuple[str, str]:
    """Returns (text, source label). Falls back to extractive text if no generator works."""
    try:
        text = gen.generate(prompt, max_tokens)
        if text:
            return text, gen.local_name
    except Exception as e:
        return _extractive(context), f"extractive fallback (generator unavailable: {type(e).__name__})"
    return _extractive(context), "extractive fallback (empty generation)"


def summarize_topic(gen, mm, topic: str) -> tuple[str, str]:
    ctx = _context([mm.topic_concepts[topic], *mm.topics[topic]])
    prompt = (f'Summarize these lecture notes about "{topic}" in 2-3 clear sentences '
              f"for a student revising. Use only the notes.\n\nNotes: {ctx}")
    return _run(gen, prompt, ctx, 120)


def explain_concept(gen, concept) -> tuple[str, str]:
    ctx = _context([concept])
    prompt = (f'Explain "{concept.name}" in simple words in 2-3 sentences. '
              f"Use only the notes.\n\nNotes: {ctx}")
    return _run(gen, prompt, ctx, 120)
