from fraud_agent.schemas import FraudDecision

# Minimum hours needed to physically travel between cities far apart.
# 3 hours between any two major international cities is impossible by commercial flight.
_IMPOSSIBLE_TRAVEL_HOURS = 4

_AMOUNT_SPIKE_MULTIPLIER = 5.0

_BLOCK_THRESHOLD = 0.65
_CHALLENGE_THRESHOLD = 0.35


def compute_fraud_score(
    transaction_amount: float,
    spend_history: dict,
    travel_signals: dict,
    device_fingerprint: dict,
) -> FraudDecision:
    score = 0.0
    reasons = []

    # Signal 1: amount spike vs historical average
    avg = spend_history.get("avg_amount", 0)
    if avg > 0 and transaction_amount > avg * _AMOUNT_SPIKE_MULTIPLIER:
        score += 0.35
        reasons.append(
            f"Amount ${transaction_amount:.0f} is "
            f"{transaction_amount / avg:.0f}x the average ${avg:.0f}"
        )

    # Signal 2: physically impossible travel
    hours_gap = travel_signals.get("hours_gap")
    last_city = travel_signals.get("last_swiped_location", "")
    current_city = travel_signals.get("current_location", "")
    if (
        hours_gap is not None
        and hours_gap < _IMPOSSIBLE_TRAVEL_HOURS
        and last_city != current_city
    ):
        score += 0.40
        reasons.append(
            f"Impossible travel: {last_city} → {current_city} in {hours_gap}h"
        )

    # Signal 3: unknown device
    if not device_fingerprint.get("known_device", True):
        score += 0.25
        reasons.append(
            f"Transaction from unrecognized device: {device_fingerprint.get('device_id')}"
        )

    # Map score to decision
    if score >= _BLOCK_THRESHOLD:
        decision = "block"
    elif score >= _CHALLENGE_THRESHOLD:
        decision = "challenge"
    else:
        decision = "approve"

    return FraudDecision(
        decision=decision,
        confidence=round(score, 2),
        reasoning="; ".join(reasons) if reasons else "No fraud signals detected",
    )
