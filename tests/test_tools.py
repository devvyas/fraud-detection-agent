from fraud_agent.tools import get_spend_history, get_travel_signals, get_device_fingerprint


def test_get_spend_history_returns_correct_shape():
    result = get_spend_history.invoke({"card_id": "4242"})

    # Verify all three fields are present
    assert "min_amount" in result
    assert "max_amount" in result
    assert "avg_amount" in result

    # Verify types — a float check catches bugs where mock returns "25.0" (string) instead of 25.0
    assert isinstance(result["min_amount"], float)
    assert isinstance(result["max_amount"], float)
    assert isinstance(result["avg_amount"], float)

    # Verify the values make logical sense
    assert result["min_amount"] <= result["avg_amount"] <= result["max_amount"]


def test_get_travel_signals_returns_correct_shape():
    result = get_travel_signals.invoke({"card_id": "4242"})

    assert "last_swiped_location" in result
    assert "current_location" in result
    assert "hours_gap" in result

    assert isinstance(result["last_swiped_location"], str)
    assert isinstance(result["current_location"], str)
    assert isinstance(result["hours_gap"], int)

    # For card 4242 the locations must be different — same city means no travel anomaly
    assert result["last_swiped_location"] != result["current_location"]


def test_get_device_fingerprint_returns_correct_shape():
    result = get_device_fingerprint.invoke({"card_id": "4242"})

    assert "device_id" in result
    assert "known_device" in result

    assert isinstance(result["device_id"], str)
    assert isinstance(result["known_device"], bool)
