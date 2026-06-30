from fraud_agent.scoring import compute_fraud_score


# Fixtures — reusable mock data representing the suspicious card 4242
SUSPICIOUS_SPEND = {"min_amount": 8.0, "max_amount": 800.0, "avg_amount": 25.0}
IMPOSSIBLE_TRAVEL = {"last_swiped_location": "New York", "current_location": "Tokyo", "hours_gap": 3}
UNKNOWN_DEVICE = {"device_id": "Apple_17", "known_device": False}
KNOWN_DEVICE = {"device_id": "Apple_17", "known_device": True}
NORMAL_TRAVEL = {"last_swiped_location": "New York", "current_location": "New York", "hours_gap": 1}


def test_all_signals_fire_produces_block():
    decision = compute_fraud_score(
        transaction_amount=5000.0,
        spend_history=SUSPICIOUS_SPEND,
        travel_signals=IMPOSSIBLE_TRAVEL,
        device_fingerprint=UNKNOWN_DEVICE,
    )
    assert decision.decision == "block"
    assert decision.confidence >= 0.65
    assert "impossible travel" in decision.reasoning.lower() or "Impossible" in decision.reasoning


def test_no_signals_produces_approve():
    decision = compute_fraud_score(
        transaction_amount=20.0,      # within normal range
        spend_history=SUSPICIOUS_SPEND,
        travel_signals=NORMAL_TRAVEL,
        device_fingerprint=KNOWN_DEVICE,
    )
    assert decision.decision == "approve"
    assert decision.confidence == 0.0


def test_only_amount_spike_produces_challenge():
    decision = compute_fraud_score(
        transaction_amount=5000.0,    # 200x average → +0.35, below block threshold
        spend_history=SUSPICIOUS_SPEND,
        travel_signals=NORMAL_TRAVEL,
        device_fingerprint=KNOWN_DEVICE,
    )
    assert decision.decision == "challenge"
    assert decision.confidence == 0.35


def test_reasoning_is_never_empty():
    decision = compute_fraud_score(
        transaction_amount=20.0,
        spend_history=SUSPICIOUS_SPEND,
        travel_signals=NORMAL_TRAVEL,
        device_fingerprint=KNOWN_DEVICE,
    )
    assert decision.reasoning != ""
