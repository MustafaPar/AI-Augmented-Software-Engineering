"""RAG — policy-grounded answers.

Task: answer a customer question using ONLY the retrieved help-center text. If the
answer isn't in the docs, the model must say it doesn't know (don't hallucinate).

Fill in YOUR_SYSTEM_PROMPT and YOUR_RETRIEVER.
Run:  python rag.py
"""
import os

from ollama import chat

MODEL = "llama3.1:8b"
DATA = os.path.join(os.path.dirname(__file__), "data", "help_center.txt")


def load_chunks() -> list[str]:
    with open(DATA, encoding="utf-8") as f:
        # one chunk per non-empty line
        return [line.strip() for line in f if line.strip()]


# TODO: return the subset of chunks relevant to `question` (simple keyword overlap is fine).
def YOUR_RETRIEVER(question: str, chunks: list[str]) -> list[str]:
    stopwords = {
        "a", "an", "the", "do", "does", "did", "i", "you", "your", "my", "me", "we", "us",
        "is", "are", "was", "to", "of", "for", "in", "on", "at", "and", "or", "it", "can",
        "how", "what", "whats", "when", "where", "why", "who", "which", "have", "has",
        "long", "with", "be", "s",
    }

    def _keywords(text: str) -> set[str]:
        words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
        return {w for w in words if w not in stopwords and len(w) >= 3}

    def _related(a: str, b: str) -> bool:
        # Prefix match handles simple word forms: ship/shipping, return/returns,
        # international/internationally.
        short, long_ = sorted((a, b), key=len)
        return len(short) >= 4 and long_.startswith(short)

    q_words = _keywords(question)
    scored = []
    for chunk in chunks:
        c_words = _keywords(chunk)
        score = sum(any(_related(q, c) for c in c_words) for q in q_words)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored[:3]]


# TODO: Fill this in! Force the model to answer only from the provided context.
YOUR_SYSTEM_PROMPT = """You are a customer-support assistant for an online store.
Answer the customer's question using ONLY the information in the Context section of their message.
Do not use outside knowledge, and do not guess or invent details.
If the Context is empty or does not contain the answer, reply exactly:
I don't know based on our help center information.
When the answer is in the Context, reply in one or two short sentences and include the specific facts (numbers, time periods, places) from the Context."""

CASES = [
    ("How long do I have to return an item?", ["30 days"]),
    ("Do you ship internationally?", ["international"]),
    ("What's your CEO's home address?", ["don't know", "do not know", "not sure", "cannot"]),
]


def answer(question: str) -> str:
    context = "\n".join(YOUR_RETRIEVER(question, load_chunks()))
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": YOUR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for question, needles in CASES:
        got = answer(question)
        ok = any(n.lower() in got for n in needles)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question!r} -> {got[:80]!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
