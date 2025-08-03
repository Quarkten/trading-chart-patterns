from typing import List
from src.data_structures import Candle


def _is_doji(candle: Candle) -> bool:
    """
    Identifies a high-confidence Doji pattern.

    A Doji is a candle where the open and close prices are very close, indicating
    indecision in the market.

    Rules:
    1.  The body size is less than 5% of the total candle range (high - low).
    2.  The total range is not unusually small, to avoid flagging flat lines
        or periods of no trading.
    """
    total_range = candle.high - candle.low
    if total_range == 0: return False

    is_doji_shape = candle.body_size / total_range < 0.05
    is_significant = total_range > (candle.open * 0.01) # Range must be >1% of open price

    return is_doji_shape and is_significant

def _is_hammer(candle: Candle, prev_candles: List[Candle]) -> bool:
    """
    Identifies a high-confidence Hammer pattern.

    A Hammer is a bullish reversal pattern that forms during a downtrend. It
    suggests that sellers drove prices down, but buyers came in to push prices
    back up toward the open.

    Rules:
    1.  There must be a preceding downtrend (e.g., 2 of the last 3 candles
        are bearish).
    2.  The lower wick is long (at least twice the size of the body).
    3.  The upper wick is very short (less than the size of the body).
    4.  The body is at the upper end of the trading range.
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

    This is a bullish reversal pattern where a small red candle is engulfed by a
    larger green candle, suggesting a shift from selling to buying pressure.

    Rules:
    1.  The previous candle must be bearish (red).
    2.  The current candle must be bullish (green).
    3.  The current candle's body must completely engulf the previous candle's body.
    4.  The current candle's body should be significantly larger to confirm the
        strength of the reversal.
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

    This is a bearish reversal pattern where a small green candle is engulfed by a
    larger red candle, suggesting a shift from buying to selling pressure.

    Rules:
    1.  The previous candle must be bullish (green).
    2.  The current candle must be bearish (red).
    3.  The current candle's body must completely engulf the previous candle's body.
    4.  The current candle's body should be significantly larger to confirm the
        strength of the reversal.
    """
    if not (prev.is_bullish and current.is_bearish):
        return False

    engulfs = current.open > prev.open and current.close < prev.close
    is_significant = current.body_size > prev.body_size * 1.2

    return engulfs and is_significant

def _is_marubozu(candle: Candle) -> bool:
    """
    Identifies a Marubozu candle.

    A Marubozu is a candle with no wicks, indicating strong, one-sided momentum.
    A bullish Marubozu opens at the low and closes at the high, while a bearish
    Marubozu opens at the high and closes at the low.

    Rule:
    1.  The body size is greater than 95% of the total candle range.
    """
    total_range = candle.high - candle.low
    if total_range == 0: return False
    return candle.body_size / total_range > 0.95

def _is_tweezer_top(current: Candle, prev: Candle) -> bool:
    """
    Identifies a Tweezer Top pattern.

    A Tweezer Top is a bearish reversal pattern that occurs at the top of an
    uptrend. It consists of two candles with nearly identical highs.

    Rules:
    1.  The first candle should be bullish.
    2.  The two candles have nearly matching highs (within a small tolerance).
    """
    if prev.is_bearish: return False # First candle should be bullish in a Tweezer Top
    tolerance = prev.high * 0.001
    return abs(prev.high - current.high) < tolerance

def _is_tweezer_bottom(current: Candle, prev: Candle) -> bool:
    """
    Identifies a Tweezer Bottom pattern.

    A Tweezer Bottom is a bullish reversal pattern that occurs at the bottom of
    a downtrend. It consists of two candles with nearly identical lows.

    Rules:
    1.  The first candle should be bearish.
    2.  The two candles have nearly matching lows (within a small tolerance).
    """
    if prev.is_bullish: return False # First candle should be bearish in a Tweezer Bottom
    tolerance = prev.low * 0.001
    return abs(prev.low - current.low) < tolerance

def _is_morning_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
    """
    Identifies a Morning Star pattern.

    A Morning Star is a bullish reversal pattern consisting of three candles:
    a long bearish candle, a small-bodied candle that gaps down, and a long
    bullish candle.

    Rules:
    1.  The first candle is a long bearish candle.
    2.  The second candle is a small-bodied candle (like a Doji or spinning
        top) that gaps down from the first candle.
    3.  The third candle is a long bullish candle that closes above the
        midpoint of the first candle's body.
    """
    if not (c1.is_bearish and c3.is_bullish):
        return False

    # Rule 1: c1 is a reasonably long bearish candle
    is_c1_long = c1.body_size > (c1.open * 0.015)

    # Rule 2: c2 is a small body and it gapped down
    is_c2_small = c2.body_size < c1.body_size * 0.3
    gapped_down = c2.high < c1.low

    # Rule 3: c3 is a strong bullish candle closing well into c1's body
    closes_in_c1 = c3.close > (c1.open + c1.close) / 2

    return is_c1_long and is_c2_small and gapped_down and closes_in_c1

def _is_evening_star(c1: Candle, c2: Candle, c3: Candle) -> bool:
    """
    Identifies an Evening Star pattern.

    An Evening Star is a bearish reversal pattern consisting of three candles:
    a long bullish candle, a small-bodied candle that gaps up, and a long
    bearish candle.

    Rules:
    1.  The first candle is a long bullish candle.
    2.  The second candle is a small-bodied candle that gaps up from the first
        candle.
    3.  The third candle is a long bearish candle that closes below the
        midpoint of the first candle's body.
    """
    if not (c1.is_bullish and c3.is_bearish):
        return False

    is_c1_long = c1.body_size > (c1.open * 0.015)
    is_c2_small = c2.body_size < c1.body_size * 0.3
    gapped_up = c2.low > c1.high
    closes_in_c1 = c3.close < (c1.open + c1.close) / 2

    return is_c1_long and is_c2_small and gapped_up and closes_in_c1

def _is_three_white_soldiers(c1: Candle, c2: Candle, c3: Candle) -> bool:
    """
    Identifies a Three White Soldiers pattern.

    This is a strong bullish reversal pattern consisting of three consecutive
    long bullish candles.

    Rules:
    1.  Three consecutive long bullish candles.
    2.  Each candle opens within the body of the previous candle.
    3.  Each candle closes at a new high, above the previous candle's close.
    """
    if not (c1.is_bullish and c2.is_bullish and c3.is_bullish):
        return False

    # Rule 1: All are long candles
    is_c1_long = c1.body_size > (c1.open * 0.01)
    is_c2_long = c2.body_size > (c2.open * 0.01)
    is_c3_long = c3.body_size > (c3.open * 0.01)
    if not (is_c1_long and is_c2_long and is_c3_long):
        return False

    # Rule 2 & 3: Opens within previous body, closes at a new high
    opens_in_body_2 = c1.close > c2.open > c1.open
    closes_higher_2 = c2.close > c1.close
    opens_in_body_3 = c2.close > c3.open > c2.open
    closes_higher_3 = c3.close > c2.close

    return opens_in_body_2 and closes_higher_2 and opens_in_body_3 and closes_higher_3

def _is_three_black_crows(c1: Candle, c2: Candle, c3: Candle) -> bool:
    """
    Identifies a Three Black Crows pattern.

    This is a strong bearish reversal pattern consisting of three consecutive
    long bearish candles.

    Rules:
    1.  Three consecutive long bearish candles.
    2.  Each candle opens within the body of the previous candle.
    3.  Each candle closes at a new low, below the previous candle's close.
    """
    if not (c1.is_bearish and c2.is_bearish and c3.is_bearish):
        return False

    # Rule 1: All are long candles
    is_c1_long = c1.body_size > (c1.open * 0.01)
    is_c2_long = c2.body_size > (c2.open * 0.01)
    is_c3_long = c3.body_size > (c3.open * 0.01)
    if not (is_c1_long and is_c2_long and is_c3_long):
        return False

    # Rule 2 & 3: Opens within previous body, closes at a new low
    opens_in_body_2 = c1.open > c2.open > c1.close
    closes_lower_2 = c2.close < c1.close
    opens_in_body_3 = c2.open > c3.open > c2.close
    closes_lower_3 = c3.close < c2.close

    return opens_in_body_2 and closes_lower_2 and opens_in_body_3 and closes_lower_3

def detect_patterns(candles: List[Candle]) -> List[Candle]:
    """
    Detects candlestick patterns in a list of candles with improved confidence logic.
    """
    for c in candles:
        c.pattern = None

    for i in range(3, len(candles)):
        current_candle = candles[i]
        prev_candle = candles[i-1]
        c1, c2, c3 = candles[i-2], candles[i-1], candles[i]

        # Check for 3-candle patterns first
        if _is_morning_star(c1, c2, c3):
            c3.pattern = "Morning Star"
        elif _is_evening_star(c1, c2, c3):
            c3.pattern = "Evening Star"
        elif _is_three_white_soldiers(c1, c2, c3):
            c3.pattern = "Three White Soldiers"
        elif _is_three_black_crows(c1, c2, c3):
            c3.pattern = "Three Black Crows"
        # Then check for 2-candle and 1-candle patterns
        elif _is_marubozu(current_candle):
            current_candle.pattern = "Marubozu"
        elif _is_bullish_engulfing(current_candle, prev_candle):
            current_candle.pattern = "Bullish Engulfing"
        elif _is_bearish_engulfing(current_candle, prev_candle):
            current_candle.pattern = "Bearish Engulfing"
        elif _is_tweezer_top(current_candle, prev_candle):
            current_candle.pattern = "Tweezer Top"
        elif _is_tweezer_bottom(current_candle, prev_candle):
            current_candle.pattern = "Tweezer Bottom"
        elif _is_doji(current_candle):
            current_candle.pattern = "Doji"
        elif _is_hammer(current_candle, candles[i-3:i]):
            current_candle.pattern = "Hammer"

    return candles
