"""K-shot prompting — ticket classification.

Task: classify a customer message into one of: billing, shipping, returns,
technical, other. Use in-context EXAMPLES (k-shot) so the model outputs exactly the
label, lowercase, with no extra text.

Only edit YOUR_SYSTEM_PROMPT.
Run:  python k_shot_prompting.py
"""
from ollama import chat

MODEL = "llama3.1:8b"
NUM_RUNS = 5

# TODO: Fill this in! Include a few labeled examples (k-shot) in the system prompt.
YOUR_SYSTEM_PROMPT = """You are a support-ticket classifier for an online store.
Classify the customer's message into exactly one of these labels:
billing, shipping, returns, technical, other

Label meanings:
- billing: charges, payments, invoices, refunds of money already charged, payment methods
- shipping: delivery status, tracking, late or lost packages, shipping times
- returns: sending an item back, exchanges, return eligibility
- technical: website or app bugs, crashes, login or checkout errors
- other: anything that does not fit the labels above

Respond with ONLY the label, in lowercase, with no punctuation, quotes, or explanation.

Examples:
Message: I was billed for a subscription I cancelled last month.
Label: billing

Message: Which payment methods do you accept?
Label: billing

Message: My tracking number hasn't updated in five days.
Label: shipping

Message: How long does delivery to Canada take?
Label: shipping

Message: The jacket doesn't fit, can I exchange it for a larger size?
Label: returns

Message: How do I print a return label for my order?
Label: returns

Message: I get an error page when I click the checkout button.
Label: technical

Message: The password reset email never arrives.
Label: technical

Message: Do you have a physical store in Berlin?
Label: other
"""

CASES = [
    ("My card was charged twice for order 4412.", "billing"),
    ("Where is my package? It's been two weeks.", "shipping"),
    ("I want to send the shoes back for a refund.", "returns"),
    ("The app crashes every time I open my cart.", "technical"),
]


def classify(system_prompt: str, message: str) -> str:
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for message, expected in CASES:
        got = classify(YOUR_SYSTEM_PROMPT, message)
        ok = got == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {message!r} -> got {got!r}, expected {expected!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
