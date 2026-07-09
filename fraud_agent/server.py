from fastapi import FastAPI
from fraud_agent.schemas import TransactionEvent, FraudDecision
from fraud_agent.agent import investigate

app = FastAPI(title="Fraud Detection Agent", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/investigate", response_model=FraudDecision)
def investigate_transaction(transaction: TransactionEvent) -> FraudDecision:
    """Run a full ReAct fraud investigation for a single transaction.
    Returns a deterministic approve/challenge/block decision with reasoning."""
    return investigate(transaction)
