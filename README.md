# Fraud Detection Agent

A ReAct agent that investigates credit card transactions for fraud using LangGraph, LangChain tools, and a deterministic scoring function. The LLM reasons about which tools to call and in what order based on each observation. The final approve/challenge/block decision is made by pure Python — not the LLM.

## Architecture

```
INPUT
  └─ TransactionEvent (card_id, amount, location, transaction_time)

REACT LOOP  ← repeats until LLM says "I'm done"
  ├─ THOUGHT   → LLM explains why it's calling the next tool
  ├─ ACTION    → LLM calls one tool
  ├─ OBSERVATION → tool returns structured data
  └─ THOUGHT   → LLM interprets the result, decides next step

AFTER THE LOOP  ← runs exactly once
  └─ compute_fraud_score()
       inputs:  spend history + travel signals + device fingerprint
       outputs: FraudDecision(decision, confidence, reasoning)
```

### Why the Decision is Deterministic

The LLM's job is evidence gathering — deciding which tools to call based on what it learns. The final decision is a weighted scoring function so that:

- Same inputs always produce the same output
- Every block decision has an auditable reason
- The scoring logic can be unit tested without an LLM

### Scoring Weights

| Signal | Weight | Threshold |
|---|---|---|
| Amount > 5x historical average | +0.35 | |
| Physically impossible travel (< 4h between distant cities) | +0.40 | |
| Unknown device | +0.25 | |
| **BLOCK** | | score ≥ 0.65 |
| **CHALLENGE** | | score ≥ 0.35 |
| **APPROVE** | | score < 0.35 |

## Package Structure

```
fraud-detection-agent/
├── run.py                    # local runner — edit transaction here to test scenarios
├── pyproject.toml            # project config and dependencies
├── fraud_agent/
│   ├── schemas.py            # Pydantic models: TransactionEvent, tool outputs, FraudDecision
│   ├── tools.py              # @tool-decorated functions (one per backend signal source)
│   ├── scoring.py            # deterministic scoring function — no LLM
│   ├── graph.py              # LangGraph StateGraph: ReAct loop wiring
│   └── agent.py              # entry point: runs graph, extracts evidence, calls scorer
└── tests/
    ├── test_tools.py         # unit tests for each tool in isolation
    └── test_scoring.py       # unit tests for all scoring scenarios
```

**Why this split matters:**
- `tools.py` has zero graph logic — testable without LangGraph
- `scoring.py` has no LLM dependency — fully unit testable
- `graph.py` is the only file that touches LangGraph — swap frameworks here only
- `agent.py` is a thin orchestrator

## Tech Stack

- **Python** 3.11
- **LangGraph** — ReAct loop (`StateGraph`, `ToolNode`, conditional edges)
- **LangChain** — `@tool` decorator, message types
- **Groq** (`llama-3.3-70b-versatile`) — LLM for reasoning
- **Pydantic** v2 — input/output schema validation
- **pytest** — unit tests for tools and scoring

## Setup

```bash
# Install dependencies
pip install langgraph langchain-core langchain-groq langchain-anthropic pydantic

# Set your Groq API key (add to ~/.zshrc or ~/.bashrc)
export GROQ_API_KEY=your_key_here
```

## Running the Agent

```bash
python3 run.py
```

Edit `run.py` to test different scenarios:

```python
transaction = TransactionEvent(
    card_id="4242",       # suspicious card
    amount=5000.0,
    location="Tokyo, Japan",
    transaction_time=date.today(),
)
```

### Test Cards

| card_id | Profile |
|---|---|
| `4242` | Suspicious — low avg spender, impossible travel NYC→Tokyo, unknown device |
| `1111` | Normal high spender — $800 avg, same city, known device |
| `9999` | New card — no spend history, unknown device and location |

## Sample Output

```
============================================================
FRAUD INVESTIGATION STARTING
============================================================
Transaction: $5000.00 at Tokyo, Japan

[THOUGHT] I am calling get_spend_history to understand typical spending
patterns and determine if $5000 is unusually large.

[ACTION] Calling tool: get_spend_history({'card_id': '4242'})
[OBSERVATION] get_spend_history: {"min_amount": 8.0, "max_amount": 800.0, "avg_amount": 25.0}

[THOUGHT] The current transaction of $5000 is significantly higher than
the maximum and average amounts on this card — a potential red flag.

[ACTION] Calling tool: get_travel_signals({'card_id': '4242'})
[OBSERVATION] get_travel_signals: {"last_swiped_location": "New York", "current_location": "Tokyo", "hours_gap": 3}

[THOUGHT] The card was last used in New York and is now being used in
Tokyo with only a 3-hour gap, which is suspicious.

[ACTION] Calling tool: get_device_fingerprint({'card_id': '4242'})
[OBSERVATION] get_device_fingerprint: {"device_id": "Apple_17", "known_device": false}

[THOUGHT] Highly suspicious: unusually large amount, impossible travel
NYC→Tokyo in 3 hours, and an unrecognized device.

------------------------------------------------------------
REACT LOOP COMPLETE — running scoring function
------------------------------------------------------------

[DECISION] BLOCK
[CONFIDENCE] 1.0
[REASONING] Amount $5000 is 200x the average $25; Impossible travel: New York → Tokyo in 3h; Transaction from unrecognized device: Apple_17
============================================================
```

## Running Tests

```bash
python3 -m pytest tests/ -v
```

```
tests/test_scoring.py::test_all_signals_fire_produces_block    PASSED
tests/test_scoring.py::test_no_signals_produces_approve        PASSED
tests/test_scoring.py::test_only_amount_spike_produces_challenge PASSED
tests/test_scoring.py::test_reasoning_is_never_empty           PASSED
tests/test_tools.py::test_get_spend_history_returns_correct_shape  PASSED
tests/test_tools.py::test_get_travel_signals_returns_correct_shape PASSED
tests/test_tools.py::test_get_device_fingerprint_returns_correct_shape PASSED
7 passed in 0.58s
```

## How the ReAct Loop Works

The loop is defined in `graph.py` as a `StateGraph` with two nodes:

```
agent node (LLM) ──→ should_continue ──→ tool node ──┐
      ↑                     │                         │
      └─────────────────────┘←────────────────────────┘
                  (loops back after each tool)
```

- **`agent` node** — sends the full message history to the LLM, gets back either a tool call or a final text response
- **`tool` node** — runs whichever tool the LLM picked and appends the result as a `ToolMessage`
- **`should_continue`** — if the last LLM message contains tool calls → go to tools; otherwise → END

`State` is a list of messages that grows with each step (`add_messages` reducer). The LLM sees the full accumulated history on every call — that's how it "remembers" what it already investigated.

## Known Limitation

New cards (`card_id="9999"`) with no spend history are currently approved despite having an unknown device. The scoring function skips the amount spike check when `avg_amount=0`. A production system should treat missing spend history as a risk signal and auto-challenge. This is left as an extension exercise.

## Extension Ideas

- **Fix the new-card bug** — add a `missing_history` signal in `scoring.py` that challenges when `avg_amount == 0`
- **Add a merchant risk tool** — flag high-risk merchant category codes (electronics, crypto exchanges)
- **Make tools async** — replace sync mock lookups with `async def` and `asyncio.gather()` for parallel evidence gathering without parallel LLM tool calls
- **Add retry logic** — if a tool returns an empty dict, retry once before scoring with partial evidence
