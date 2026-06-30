from datetime import date
from fraud_agent.schemas import TransactionEvent
from fraud_agent.agent import investigate

# Edit this transaction to test different scenarios
transaction = TransactionEvent(
    card_id="4242",
    amount=5000.0,
    location="Tokyo, Japan",
    transaction_time=date.today(),
)

investigate(transaction)
