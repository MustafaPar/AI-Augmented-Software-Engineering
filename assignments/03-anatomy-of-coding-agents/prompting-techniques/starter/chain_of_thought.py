"""Chain-of-thought — refund eligibility.

Task: given the refund policy and an order, decide YES or NO. Use chain-of-thought
so the model reasons step by step, but make the FINAL line exactly `DECISION: YES`
or `DECISION: NO` so it can be graded.

Only edit YOUR_SYSTEM_PROMPT.
Run:  python chain_of_thought.py
"""
from ollama import chat

MODEL = "llama3.1:8b"

POLICY = """Refund policy:
- Refunds allowed within 30 days of purchase.
- Item must be unused / in original condition.
- Final-sale items are never refundable.
"""

# TODO: Fill this in! Encourage step-by-step reasoning, ending with DECISION: YES/NO.
YOUR_SYSTEM_PROMPT = """You are a support agent who decides whether an order is eligible for a refund.
Reason step by step, checking EVERY rule of the refund policy below in order:
Step 1 - Time: Was it purchased within 30 days? (More than 30 days means NOT eligible.)
Step 2 - Condition: Is the item unused / in original condition? (If the customer does not mention any use or damage, assume it is unused.)
Step 3 - Final sale: Is it a final-sale item? (If it is final sale, it is NOT eligible. If not mentioned, assume it is not final sale.)
Step 4 - Conclusion: The order is eligible only if it passes ALL rules. If any single rule fails, it is not eligible.

Write one short line for each step. Then write the final line exactly as either:
DECISION: YES
or
DECISION: NO
Nothing may come after the DECISION line.
"""

CASES = [
    ("Bought 10 days ago, unused, not final sale.", "YES"),
    ("Bought 45 days ago, unused.", "NO"),
    ("Bought 5 days ago but it was a final-sale item.", "NO"),
]


def decide(system_prompt: str, order: str) -> str:
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt + "\n" + POLICY},
            {"role": "user", "content": order},
        ],
        options={"temperature": 0.0},
    )
    text = resp.message.content.strip()
    last = text.splitlines()[-1].upper() if text else ""
    return "YES" if "DECISION: YES" in last else "NO" if "DECISION: NO" in last else "?"


def run() -> bool:
    passed = 0
    for order, expected in CASES:
        got = decide(YOUR_SYSTEM_PROMPT, order)
        ok = got == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {order!r} -> {got}, expected {expected}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
