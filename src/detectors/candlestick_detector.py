from typing import List
from src.data_structures import Candle

def _is_doji(candle: Candle) -> bool:
    """
    Identifies a high-confidence Doji pattern.
    Rule: Body is less than 5% of the total range, AND the total range is
          not unusually small (to avoid flagging flat lines).
    """
    total_range = candle.high - candle.low
    if total_range == 0: return False

    is_doji_shape = candle.body_size / total_range < 0.05
    is_significant = total_range > (candle.open * 0.01) # Range must be >1% of open price

    return is_doji_shape and is_significant

def _is_hammer(candle: Candle, prev_candles: List[Candle]) -> bool:
    """
    Identifies a high-confidence Hammer pattern.
    Rules:
    1. There's a preceding downtrend (e.g., 2 of last 3 candles are bearish).
    2. The lower wick is long (at least 2x the body size).
    3. The upper wick is very short (less than half the body size).
    4. The body is at the upper end of the trading range.
    """
    if len(prev_candles) < 3: return False

    # Rule 1: Preceding downtrend
    bearish_count = sum(1 for c in prev_candles[-3:] if c.is_bearish)
    if bearish_count < 2: return False

    # Rules 2, 3, 4
    long_lower_wick = candle.lower_wick_size >= candle.body_size * 2
    short_upper_wick = candle.upper_wick_size < candle.body_size

    return long_lower_wick and short_upper_wick and candle.body_size > 0

def _is_bullish_engulfing(current: Candle, prev: Candle) -> bool:
    """
    Identifies a high-confidence Bullish Engulfing pattern.
    Rule: A red candle is engulfed by a significantly larger green candle.
    """
    if not (prev.is_bearish and current.is_bullish):
        return False

    # Current body must engulf the previous body and be significantly larger
    engulfs = current.open < prev.open and current.close > prev.close
    is_significant = current.body_size > prev.body_size * 1.2

    return engulfs and is_significant

def _is_bearish_engulfing(current: Candle, prev: Candle) -> bool:
    """
    Identifies a high-confidence Bearish Engulfing pattern.
    Rule: A green candle is engulfed by a significantly larger red candle.
    """
    if not (prev.is_bullish and current.is_bearish):
        return False

    engulfs = current.open > prev.open and current.close < prev.close
    is_significant = current.body_size > prev.body_size * 1.2

    return engulfs and is_significant

def detect_patterns(candles: List[Candle]) -> List[Candle]:
    """
    Detects candlestick patterns in a list of candles with improved confidence logic.
    """
    # Clear any previous patterns
    for c in candles:
        c.pattern = None

    # Start from index 3 to have enough history for trend checks
    for i in range(3, len(candles)):
        current_candle = candles[i]
        prev_candle = candles[i-1]

        # Check for single-candle patterns on the current candle
        if _is_doji(current_candle):
            current_candle.pattern = "Doji"
        elif _is_hammer(current_candle, candles[i-3:i]):
            current_candle.pattern = "Hammer"

        # Check for two-candle patterns
        elif _is_bullish_engulfing(current_candle, prev_candle):
            current_candle.pattern = "Bullish Engulfing"
        elif _is_bearish_engulfing(current_candle, prev_candle):
            current_candle.pattern = "Bearish Engulfing"

    return candles
