import matplotlib.pyplot as plt
import io
import numpy as np
import cv2
from src.data_structures import Candle

def generate_chart_image(candles: list[Candle], ticker: str, img_size=(640, 640)) -> np.ndarray:
    """
    Generates a candlestick chart image in memory suitable for YOLOv8 inference.

    Args:
        candles: A list of Candle objects representing the stock data.
        ticker: The stock ticker symbol.
        img_size: The desired output image size (width, height).

    Returns:
        A NumPy array representing the chart image (BGR format).
    """
    fig, ax = plt.subplots(figsize=(img_size / 100, img_size / 100), dpi=100)
    ax.set_title(f"{ticker} Stock Chart")
    ax.set_facecolor('lightgray')

    for candle in candles:
        color = 'green' if candle.is_bullish else 'red'
        ax.plot([candle.index, candle.index], [candle.low, candle.high], color=color, linewidth=1)
        body = plt.Rectangle((candle.index - 0.4, candle.open), 0.8, candle.close - candle.open, color=color)
        ax.add_patch(body)

    ax.autoscale_view()
    ax.axis('off') # Hide axes for cleaner chart image

    # Save the chart to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    # Read the image from buffer using OpenCV
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # Reads as BGR

    # Resize to the target size
    img = cv2.resize(img, img_size)

    return img

if __name__ == "__main__":
    # Example usage (for testing purposes)
    from datetime import date, timedelta
    from polygon import RESTClient
    import os

    # Dummy Candle class for testing if src.data_structures is not available
    class Candle:
        def __init__(self, index, open, high, low, close):
            self.index = index
            self.open = open
            self.high = high
            self.low = low
            self.close = close
            self.is_bullish = close > open
            self.pattern = None # No pattern detection in this module

    def get_polygon_client_test() -> RESTClient:
        api_key = os.environ.get("POLYGON_API_KEY", "YOUR_POLYGON_API_KEY")
        if api_key == "YOUR_POLYGON_API_KEY":
            print("WARNING: Using a placeholder API key. Please set POLYGON_API_KEY environment variable for real data.")
        return RESTClient(api_key)

    def fetch_stock_data_test(client: RESTClient, ticker: str, start_date: str, end_date: str) -> list:
        try:
            aggs = client.get_aggs(
                ticker=ticker, multiplier=1, timespan="day",
                from_=start_date, to=end_date, limit=50000
            )
            print(f"Successfully fetched {len(aggs)} data points for {ticker}.")
            return aggs
        except Exception as e:
            print(f"Could not fetch data for {ticker}: {e}")
            return []

    client = get_polygon_client_test()
    ticker = "AAPL"
    end_date = date.today()
    start_date = end_date - timedelta(days=30) # Fetch last 30 days

    data = fetch_stock_data_test(client, ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if data:
        test_candles = [Candle(i, agg.open, agg.high, agg.low, agg.close) for i, agg in enumerate(data)]
        chart_image = generate_chart_image(test_candles, ticker)
        cv2.imwrite(f"test_chart_{ticker}.png", chart_image)
        print(f"Test chart saved to test_chart_{ticker}.png")
    else:
        print("No data fetched for testing chart generation.")