# Week 3 — Prompting Techniques: Writeup

All runs used the local Ollama model `llama3.1:8b` at `temperature=0.0`.
All sample outputs below are copied from real runs.

| Technique | Check result |
|---|---|
| K-shot prompting | 4/4 PASS |
| Chain-of-thought | 3/3 PASS |
| Tool calling | PASS (`order_id='4412'`) |
| RAG | 3/3 PASS |

To measure what each prompt contributed, I also ran every task once with an
**empty system prompt** (the baseline). Those baseline results are summarized in each section.

---

## 1. K-shot prompting — ticket classification (`k_shot_prompting.py`)

**Final prompt**
```text
You are a support-ticket classifier for an online store.
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
```
The examples are different from the test cases, so the model has to generalize
instead of copying memorized answers.

**Sample input / output**
- Input: `My card was charged twice for order 4412.`
- Output: `billing`

**Baseline (empty prompt):** 0/4. The model answered like a chatbot instead of classifying,
e.g. `"i'm so sorry to hear that your card was charged twice for order 4412. i'd be happy to help…"`.

**What I learned:** The model already knew the domain; the few labeled examples mostly
taught it the output *format*, changing chatty paragraphs into exactly one lowercase label (0/4 → 4/4).

---

## 2. Chain-of-thought — refund eligibility (`chain_of_thought.py`)

**Final prompt** (the starter code appends the refund policy after it)
```text
You are a support agent who decides whether an order is eligible for a refund.
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
```

**Sample input / output**
- Input: `Bought 5 days ago but it was a final-sale item.`
- Output:
  ```text
  Step 1 - Time: Yes, it was purchased within 30 days.
  Step 2 - Condition: Since the customer does not mention any use or damage, assume it is unused.
  Step 3 - Final sale: Yes, it is a final-sale item.
  Step 4 - Conclusion: The order is not eligible because it fails the final sale rule.

  DECISION: NO
  ```

**Baseline (empty prompt):** 0/3 were gradeable (`?`). The model reasoned in prose but never wrote
a `DECISION:` line. For the 45-day case it even made up a store-credit offer that isn't in the policy.

**What I learned:** Turning each policy rule into a numbered step made the model check
every rule (including the "passes once, fails on final sale" case) and put the verdict
in a fixed, parseable final line, so the reasoning became both visible and gradeable.

---

## 3. Tool calling — order lookup (`tool_calling.py`)

**Final prompt**
```text
You are a support-desk assistant for an online store.
You cannot see order information yourself. Whenever the customer mentions an order or asks about its status or delivery, you MUST first call the get_order_status tool with the order number from their message (digits only) before replying.
Never guess or invent an order status. After the tool returns, give the customer a short, friendly reply that states the status.
```
**Tool schema descriptions**
- `get_order_status`: *"Look up the current status of a customer's order (e.g. processing, shipped, delivery date). Call this whenever a customer asks about an order, its status, or its delivery."*
- `order_id`: *"The order number exactly as written in the customer's message, digits only, e.g. '4412'."*

**Sample input / output** (complete loop: the starter check stops after the tool call, so I
sent the tool result back to the model for the final reply)
- Input: `Hi, can you check on order 4412 for me?`
- Tool call: `get_order_status({'order_id': '4412'})` → `shipped, arriving Friday`
- Final reply: `Your order 4412 has shipped and is expected to arrive on Friday. Is there anything else I can help you with?`

**Baseline (empty prompt and empty descriptions):** The model *still* called the tool correctly
with `order_id='4412'`, because this request clearly matches the only tool and its name explains itself.

**What I learned:** The tool schema alone was enough for the model to pick the tool and extract
the right ID; the model only *requests* the call, so the application has to run it and send
the result back for the final answer.

---

## 4. RAG — policy-grounded answers (`rag.py`)

**Final prompt**
```text
You are a customer-support assistant for an online store.
Answer the customer's question using ONLY the information in the Context section of their message.
Do not use outside knowledge, and do not guess or invent details.
If the Context is empty or does not contain the answer, reply exactly:
I don't know based on our help center information.
When the answer is in the Context, reply in one or two short sentences and include the specific facts (numbers, time periods, places) from the Context.
```

**Retriever:** keyword overlap over `data/help_center.txt` (the only knowledge source). It
lowercases the text, removes stopwords, and counts a match when one word is a prefix of the
other (≥4 chars), so `ship` matches `shipping` and `international` matches `internationally`.
It returns up to the top 3 lines with a score above 0, and nothing when no keyword matches.

**Sample input / output**
- Input: `Do you ship internationally?`
- Retrieved context:
  `Shipping: We offer international shipping to most countries; delivery takes 7-21 business days.`,
  `Shipping: Standard domestic shipping takes 3-5 business days.`,
  `Billing: We accept Visa, Mastercard, and PayPal. You are charged when your order ships.`
- Output: `yes, we offer international shipping to most countries.`

Out-of-scope input: `What's your CEO's home address?` → retrieved context `[]` →
`i don't know based on our help center information.`

**Baseline (empty prompt, empty context) for the CEO question:** `I can’t provide that information. …`
That is a reasonable refusal, but it uses a curly apostrophe and none of the phrases the check looks for,
so it would FAIL.

**What I learned:** Grounding needs two pieces: a retriever that returns *nothing* when nothing
matches, and a prompt that defines one exact fallback sentence, so a refusal becomes predictable
and testable instead of whatever wording the model happens to choose.

---

## Closing: why LLMs handle structured tasks well, and where they need scaffolding

Across all four tasks, `llama3.1:8b` already had the needed knowledge. With empty prompts it
understood the tickets, applied the refund rules mostly correctly, picked the right tool, and
declined the off-topic question. LLMs are good at structured tasks because classifying,
following rules, filling in arguments and summarizing text are patterns they saw constantly
in training. What failed without scaffolding was the **interface**: output that a program can parse
(a single label, a final `DECISION:` line, an exact refusal sentence), every rule checked instead of
a fluent shortcut, and facts restricted to what we actually provided rather than made-up extras like
the store credit. Prompting techniques are that scaffolding. Examples set the format,
step-by-step structure enforces procedure, tool schemas connect the model to real data, and retrieval
plus an explicit fallback limits what it is allowed to claim. Even then, the surrounding code still
does the work the model can't: running the tool, choosing what context to retrieve, and checking the output.
That division of work is the core design of a coding agent.
