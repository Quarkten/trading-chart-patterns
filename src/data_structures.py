from dataclasses import dataclass
from typing import Optional

@dataclass
class Candle:
    """
    Represents a single candlestick with OHLC data.
    """
    index: int
    open: float
    high: float
    low: float
    close: float
    pattern: Optional[str] = None

    @property
    def is_bullish(self) -> bool:
        """A candle is bullish if the close is higher than the open."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """A candle is bearish if the close is lower than the open."""
        return self.close < self.open

    @property
    def body_size(self) -> float:
        """The size of the candle's body."""
        return abs(self.open - self.close)

    @property
    def upper_wick_size(self) -> float:
        """The size of the upper wick."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick_size(self) -> float:
        """The size of the lower wick."""
        return min(self.open, self.close) - self.low
