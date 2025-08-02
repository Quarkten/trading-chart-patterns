from typing import List
from src.data_structures import Candle

def get_sample_candles() -> List[Candle]:
    """
    Provides a hardcoded list of sample candlestick data for testing.

    The list includes examples of normal candles, a Doji, and a Hammer.
    """
    sample_candles = [
        # Normal bullish candle
        Candle(index=0, open=100, high=105, low=98, close=104),

        # Normal bearish candle
        Candle(index=1, open=104, high=106, low=100, close=101),

        # A Doji pattern: open and close are very close
        Candle(index=2, open=101, high=105, low=97, close=101.1),

        # Another normal bullish candle
        Candle(index=3, open=101.1, high=108, low=100, close=107),

        # A Hammer pattern: small body, long lower wick, short upper wick
        Candle(index=4, open=102, high=103, low=95, close=102.5),

        # A normal candle to precede the engulfing pattern
        Candle(index=5, open=102.5, high=105, low=101, close=102), # Small red candle

        # A Bullish Engulfing candle
        Candle(index=6, open=101, high=108, low=100, close=107), # Large green candle that engulfs the previous one

        # Another normal candle
        Candle(index=7, open=107, high=110, low=106, close=109), # Small green candle

        # A Bearish Engulfing candle
        Candle(index=8, open=110, high=111, low=105, close=106), # Large red candle that engulfs the previous one
    ]
    return sample_candles
