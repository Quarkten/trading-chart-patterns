import os
from datetime import datetime, timedelta
import pandas as pd
from polygon import RESTClient
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class DataFetcher:
    def __init__(self):
        self.api_key = os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set.")
        self.client = RESTClient(self.api_key)

    def get_candlestick_data(self, symbol: str, multiplier: int, timespan: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Fetches candlestick data from Polygon.io.

        :param symbol: Stock ticker symbol (e.g., "AAPL").
        :param multiplier: The size of the timespan multiplier.
        :param timespan: The size of the time period. (e.g., "minute", "hour", "day", "week", "month", "quarter", "year").
        :param from_date: Start date in YYYY-MM-DD format.
        :param to_date: End date in YYYY-MM-DD format.
        :return: Pandas DataFrame with OHLCV data, or empty DataFrame if no data.
        """
        try:
            aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date,
                to=to_date,
                raw=False
            )

            if not aggs:
                print(f"No data found for {symbol} from {from_date} to {to_date} with timeframe {timeframe}.")
                return pd.DataFrame()

            data = []
            for agg in aggs:
                data.append({
                    'timestamp': datetime.fromtimestamp(agg.timestamp / 1000), # Convert ms to s
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

if __name__ == '__main__':
    # Example Usage:
    fetcher = DataFetcher()
    
    # Fetch 15-minute data for AAPL for the last 7 days
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    print(f"Fetching AAPL 15-minute data from {from_date} to {to_date}...")
    aapl_data = fetcher.get_candlestick_data("AAPL", 15, "minute", from_date, to_date)
    print(aapl_data.head())

    # Fetch 5-minute data for MSFT for a specific day
    print("\nFetching MSFT 5-minute data for 2024-07-20...")
    msft_data = fetcher.get_candlestick_data("MSFT", 5, "minute", "2024-07-20", "2024-07-20")
    print(msft_data.head())

