from pydantic import BaseModel
from datetime import date
from typing import Literal

class TransactionEvent(BaseModel):
    card_id: str
    amount: float
    location: str
    transaction_time: date

class SpendHistoryResult(BaseModel):
    min_amount: float
    max_amount: float
    avg_amount: float

class TravelSignalResult(BaseModel):
    last_swiped_location: str
    current_location: str
    hours_gap: int

class DeviceFingerprintResult(BaseModel):
    device_id: str
    known_device: bool 

class FraudDecision(BaseModel):
    decision: Literal["approve","challenge","block"]
    confidence: float
    reasoning: str