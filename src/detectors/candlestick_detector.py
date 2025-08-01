from typing import List
from src.data_structures import Candle

def _is_doji(candle: Candle) -> bool:
    """
    Identifies a Doji pattern.
    A Doji is a candle where the body is very small compared to the wicks.
    Rule: The body size is less than 5% of the total candle range (high - low).
    """
    total_range = candle.high - candle.low
    if total_range == 0:
        return False  # Avoid division by zero for flat candles
    return candle.body_size / total_range < 0.05

def _is_hammer(candle: Candle) -> bool:
    """
    Identifies a Hammer pattern.
    A Hammer is a bullish reversal pattern that forms during a downtrend.
    Rules:
    1. The candle is bullish (or has a very small bearish body).
    2. The lower wick is long (at least 2x the body size).
    3. The upper wick is very short or non-existent.
    """
    # For simplicity, we are not checking for a preceding downtrend here.
    # This check would require analyzing the previous candles.

    # Rule 2: Long lower wick
    long_lower_wick = candle.lower_wick_size > candle.body_size * 2

    # Rule 3: Short upper wick (e.g., less than 20% of the lower wick's size)
    # This makes the rule relative to the most prominent feature of the candle.
    short_upper_wick = candle.upper_wick_size < candle.lower_wick_size * 0.2

    return long_lower_wick and short_upper_wick

def detect_patterns(candles: List[Candle]) -> List[Candle]:
    """
    Detects candlestick patterns in a list of candles.

    This function iterates through the list of candles and checks for known
    patterns. If a pattern is detected, the `pattern` attribute of the
    candle is updated with the pattern's name.

    Args:
        candles: A list of Candle objects.

    Returns:
        The same list of Candle objects with the `pattern` attribute updated.
    """
    for candle in candles:
        # Check for each pattern. Add more checks here as more patterns are implemented.
        if _is_doji(candle):
            candle.pattern = "Doji"
        elif _is_hammer(candle):
            candle.pattern = "Hammer"

    return candles
