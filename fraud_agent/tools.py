from langchain_core.tools import tool
from fraud_agent.schemas import SpendHistoryResult, TravelSignalResult, DeviceFingerprintResult

# ---------------------------------------------------------------------------
# Simulated data store — in production each tool would call a different
# backend: a spend DB, a geolocation service, a device vault.
# The tool interface (card_id in, dict out) stays identical either way.
# ---------------------------------------------------------------------------

_SPEND_DB = {
    "4242": SpendHistoryResult(min_amount=8.0,   max_amount=800.0,  avg_amount=25.0),
    "1111": SpendHistoryResult(min_amount=50.0,  max_amount=5000.0, avg_amount=800.0),
    "9999": SpendHistoryResult(min_amount=0.0,   max_amount=0.0,    avg_amount=0.0),   # new card, no history
}

_TRAVEL_DB = {
    "4242": TravelSignalResult(last_swiped_location="New York", current_location="Tokyo",    hours_gap=3),
    "1111": TravelSignalResult(last_swiped_location="New York", current_location="New York", hours_gap=2),
    "9999": TravelSignalResult(last_swiped_location="Unknown",  current_location="Unknown",  hours_gap=0),
}

_DEVICE_DB = {
    "4242": DeviceFingerprintResult(device_id="Apple_17",   known_device=False),
    "1111": DeviceFingerprintResult(device_id="Samsung_S24", known_device=True),
    "9999": DeviceFingerprintResult(device_id="Unknown",     known_device=False),
}

_DEFAULT_SPEND  = SpendHistoryResult(min_amount=0.0, max_amount=0.0, avg_amount=0.0)
_DEFAULT_TRAVEL = TravelSignalResult(last_swiped_location="Unknown", current_location="Unknown", hours_gap=0)
_DEFAULT_DEVICE = DeviceFingerprintResult(device_id="Unknown", known_device=False)


@tool
def get_spend_history(card_id: str) -> dict:
    """Use this tool to retrieve historical spending patterns for a card.
    Returns the minimum, maximum, and average transaction amounts seen on this card."""
    result = _SPEND_DB.get(card_id, _DEFAULT_SPEND)
    return result.model_dump()


@tool
def get_travel_signals(card_id: str) -> dict:
    """Use this tool to check for travel anomalies on a card.
    Returns the last known swipe location, current transaction location, and the
    hours elapsed between them. A small hours_gap between distant cities is suspicious."""
    result = _TRAVEL_DB.get(card_id, _DEFAULT_TRAVEL)
    return result.model_dump()


@tool
def get_device_fingerprint(card_id: str) -> dict:
    """Use this tool to check whether the device used for a transaction is recognized.
    Returns the device identifier and whether it is a known device for this card."""
    result = _DEVICE_DB.get(card_id, _DEFAULT_DEVICE)
    return result.model_dump()
