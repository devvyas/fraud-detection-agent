from langchain_core.tools import tool
from fraud_agent.schemas import SpendHistoryResult, TravelSignalResult, DeviceFingerprintResult


@tool
def get_spend_history(card_id: str) -> dict:
    """Use this tool to retrieve historical spending patterns for a card."""
    result = SpendHistoryResult(
        min_amount=8.0,
        max_amount=800.0,
        avg_amount=25.0
        )
    return result.model_dump()

@tool
def get_travel_signals(card_id: str) -> dict:
    """Use this tool to retrieve last known and current location of swiper card."""
    result = TravelSignalResult(
        last_swiped_location='New York',
        current_location='Tokyo',
        hours_gap=3
        )
    return result.model_dump()

@tool
def get_device_fingerprint(card_id: str) -> dict:
     
    """Use this tool to retrieve device fingerprint for a card."""
    result = DeviceFingerprintResult(
        device_id='Apple_17',
        known_device=False
        )
    return result.model_dump()