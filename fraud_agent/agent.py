from langchain_core.messages import HumanMessage, ToolMessage

from fraud_agent.graph import build_graph, _SYSTEM_PROMPT
from fraud_agent.schemas import FraudDecision, TransactionEvent
from fraud_agent.scoring import compute_fraud_score


def investigate(transaction: TransactionEvent) -> FraudDecision:
    """Run a full fraud investigation for a single transaction.

    The LLM drives tool selection. After the ReAct loop ends, a deterministic
    scoring function produces the final decision.
    """
    graph = build_graph()

    # The initial message describes the transaction in natural language.
    # The LLM reads this and decides which tools to call first.
    initial_message = HumanMessage(content=(
        f"Investigate this transaction for fraud:\n"
        f"  Card ID:    {transaction.card_id}\n"
        f"  Amount:     ${transaction.amount:.2f}\n"
        f"  Location:   {transaction.location}\n"
        f"  Time:       {transaction.transaction_time}\n\n"
        f"Use the available tools to gather evidence, then summarize your findings."
    ))

    print("\n" + "="*60)
    print("FRAUD INVESTIGATION STARTING")
    print("="*60)
    print(f"Transaction: ${transaction.amount:.2f} at {transaction.location}")

    # stream() yields each step of the graph as it executes.
    # This lets us print the Thought → Action → Observation loop in real time.
    final_state = None
    for step in graph.stream(
        {"messages": [_SYSTEM_PROMPT, initial_message]},
        stream_mode="values",
    ):
        final_state = step
        last_msg = step["messages"][-1]

        # Print what's happening at each step
        msg_type = type(last_msg).__name__
        if msg_type == "AIMessage":
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    print(f"\n[ACTION] Calling tool: {tc['name']}({tc['args']})")
            else:
                print(f"\n[THOUGHT] LLM summary:\n{last_msg.content}")
        elif msg_type == "ToolMessage":
            print(f"[OBSERVATION] {last_msg.name}: {last_msg.content[:200]}")

    print("\n" + "-"*60)
    print("REACT LOOP COMPLETE — running scoring function")
    print("-"*60)

    # Extract tool outputs from the final message history.
    # We find each ToolMessage by its tool name and parse the content.
    messages = final_state["messages"]
    tool_results: dict[str, dict] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            import json
            try:
                tool_results[msg.name] = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                tool_results[msg.name] = {}

    # Run the deterministic scoring function with whatever evidence was gathered.
    decision = compute_fraud_score(
        transaction_amount=transaction.amount,
        spend_history=tool_results.get("get_spend_history", {}),
        travel_signals=tool_results.get("get_travel_signals", {}),
        device_fingerprint=tool_results.get("get_device_fingerprint", {}),
    )

    print(f"\n[DECISION] {decision.decision.upper()}")
    print(f"[CONFIDENCE] {decision.confidence}")
    print(f"[REASONING] {decision.reasoning}")
    print("="*60 + "\n")

    return decision
