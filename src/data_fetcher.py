import os
import asyncio
import json
import websockets
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical import StockHistoricalDataClient

load_dotenv() # Load environment variables from .env file

from alpaca.data.live import StockDataStream

class DataFetcher:
    MAX_ALPACA_RETRIES = 3 # Reduced retries for quicker fallback

    def __init__(self, socketio=None):
        self.should_run = True # Initialize should_run
        self.api_key = os.environ.get("ALPACA_API_KEY")
        self.secret_key = os.environ.get("ALPACA_SECRET_KEY")
        self.finnhub_api_key = os.environ.get("FINNHUB_API_KEY")
        if not self.api_key or not self.secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables not set.")
        if not self.finnhub_api_key:
            print("Warning: FINNHUB_API_KEY environment variable not set. Finnhub real-time data will not be available.")
        self.stock_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.socketio = socketio
        self.websocket_client = StockDataStream(self.api_key, self.secret_key)

    async def _bar_data_handler(self, bar_data):
        # This handler now expects a dictionary with 'time', 'open', 'high', 'low', 'close', 'volume'
        # from either Alpaca or Finnhub.
        print(f"Real-time bar received: {bar_data}")
        if self.socketio:
            self.socketio.emit('new_bar', bar_data)

    async def _finnhub_websocket_connect(self, symbols: list):
        if not self.finnhub_api_key:
            print("Finnhub API key not set. Cannot connect to Finnhub websocket.")
            return

        finnhub_ws_url = f"wss://ws.finnhub.io?token={self.finnhub_api_key}"

        while self.should_run:
            try:
                async with websockets.connect(finnhub_ws_url) as websocket:
                    print(f"Connected to Finnhub websocket for symbols: {symbols}")
                    # Subscribe to trades for each symbol
                    for symbol in symbols:
                        await websocket.send(json.dumps({"type": "subscribe", "symbol": symbol}))
                        print(f"Subscribed to Finnhub trades for {symbol}")

                    while self.should_run:
                        message = await websocket.recv()
                        data = json.loads(message)

                        # Finnhub sends different types of messages, we are interested in 'trade' messages
                        if data.get("type") == "trade":
                            for trade in data.get("data", []):
                                # Finnhub trade data doesn't directly provide OHLCV bars.
                                # For simplicity, we'll use the last trade price as 'close'
                                # and current timestamp. This is a simplification for real-time bars.
                                # A proper bar aggregation would require more complex logic.
                                formatted_bar = {
                                    "time": trade["t"] // 1000,  # Convert ms to seconds
                                    "open": trade["p"],
                                    "high": trade["p"],
                                    "low": trade["p"],
                                    "close": trade["p"],
                                    "volume": trade["v"]
                                }
                                await self._bar_data_handler(formatted_bar) # Pass dict directly
            except websockets.exceptions.ConnectionClosedOK:
                print("Finnhub websocket connection closed gracefully.")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"Finnhub websocket connection closed with error: {e}. Reconnecting...")
                await asyncio.sleep(5) # Wait before reconnecting
            except Exception as e:
                print(f"An unexpected error occurred with Finnhub websocket: {e}. Reconnecting...")
                await asyncio.sleep(5) # Wait before reconnecting

    async def start_realtime_stream(self, symbols: list):
        print(f"Starting real-time stream for symbols: {symbols}")
        self.should_run = True
        alpaca_retry_count = 0
        # MAX_ALPACA_RETRIES = 3 # Reduced retries for quicker fallback
        BASE_DELAY = 1  # seconds

        # Try Alpaca first
        while self.should_run and alpaca_retry_count < self.MAX_ALPACA_RETRIES:
            try:
                # Re-initialize websocket client on each attempt to ensure a clean state
                self.websocket_client = StockDataStream(self.api_key, self.secret_key)
                self.websocket_client.subscribe_bars(self._alpaca_bar_data_handler, *symbols) # Use specific handler for Alpaca
                print("Attempting to connect to Alpaca WebSocket client...")
                task = asyncio.create_task(self.websocket_client.run())
                await task
                print("Alpaca WebSocket client stopped gracefully.")
                return # Exit if Alpaca stream runs successfully
            except ValueError as e:
                if "connection limit exceeded" in str(e):
                    print(f"Alpaca connection limit exceeded. Retrying Alpaca in {BASE_DELAY * (2 ** alpaca_retry_count)} seconds...")
                    alpaca_retry_count += 1
                    await asyncio.sleep(BASE_DELAY * (2 ** (alpaca_retry_count - 1)))
                else:
                    print(f"An unexpected ValueError occurred with Alpaca: {e}. Retrying Alpaca...")
                    alpaca_retry_count += 1
                    await asyncio.sleep(BASE_DELAY * (2 ** (alpaca_retry_count - 1)))
            except asyncio.CancelledError:
                print("Alpaca stream cancelled gracefully.")
                return # Exit the loop if cancelled
            except Exception as e:
                print(f"An unexpected error occurred during Alpaca websocket communication: {e}. Retrying Alpaca...")
                alpaca_retry_count += 1
                await asyncio.sleep(BASE_DELAY * (2 ** (alpaca_retry_count - 1)))
        
        if alpaca_retry_count >= self.MAX_ALPACA_RETRIES:
            print("Max Alpaca retries reached. Attempting to switch to Finnhub real-time stream.")
            # Fallback to Finnhub
            await self._finnhub_websocket_connect(symbols)

    async def stop_realtime_stream(self):
        print("Stopping real-time stream...")
        self.should_run = False
        if self.websocket_client:
            await self.websocket_client.close()
            print("Alpaca WebSocket client closed.")
        # No explicit close for Finnhub websockets as they are managed by the async with block

    async def _alpaca_bar_data_handler(self, bar):
        # Format the bar data for the frontend (similar to get_candlestick_data output)
        formatted_bar = {
            "time": bar.timestamp.timestamp(), # Unix timestamp in seconds
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        }
        await self._bar_data_handler(formatted_bar)

    def get_candlestick_data(self, symbol: str, multiplier: int, timespan: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Fetches candlestick data from Alpaca API.

        :param symbol: Stock ticker symbol (e.g., "AAPL").
        :param multiplier: The size of the timespan multiplier.
        :param timespan: The size of the time period. (e.g., "minute", "hour", "day", "month").
        :param from_date: Start date in YYYY-MM-DD format.
        :param to_date: End date in YYYY-MM-DD format.
        :return: Pandas DataFrame with OHLCV data, or empty DataFrame if no data.
        """
        try:
            # Map timespan string to Alpaca TimeFrameUnit
            timeframe_unit_map = {
                "minute": TimeFrameUnit.Minute,
                "hour": TimeFrameUnit.Hour,
                "day": TimeFrameUnit.Day,
                "month": TimeFrameUnit.Month,
            }
            unit = timeframe_unit_map.get(timespan.lower())
            if not unit:
                raise ValueError(f"Unsupported timespan: {timespan}")

            timeframe_obj = TimeFrame(multiplier, unit)

            # Convert date strings to datetime objects
            start_dt = datetime.strptime(from_date, '%Y-%m-%d')
            end_dt = datetime.strptime(to_date, '%Y-%m-%d')

            stock_request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe_obj,
                start=start_dt,
                end=end_dt
            )

            stock_bars = self.stock_client.get_stock_bars(stock_request_params)

            if not stock_bars or stock_bars.df.empty:
                print(f"No data found for {symbol} from {from_date} to {to_date} with timeframe {multiplier} {timespan}.")
                return pd.DataFrame()

            df = stock_bars.df
            # Alpaca returns data with a MultiIndex if multiple symbols are requested
            # Ensure it's a flat DataFrame with 'timestamp' as a column
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=1) # Reset symbol index
                df = df.rename(columns={'timestamp': 'timestamp'}) # Rename level_0 to timestamp
            else:
                df = df.reset_index() # Reset timestamp index

            # Ensure column names are consistent (Alpaca uses 'timestamp', 'open', 'high', 'low', 'close', 'volume')
            df = df.rename(columns={
                'timestamp': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Convert timestamp to Unix timestamp in seconds for lightweight-charts
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
            
            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

if __name__ == '__main__':
    # Example Usage:
    fetcher = DataFetcher()
    
    # Set dummy API keys for example if not in .env
    os.environ["ALPACA_API_KEY"] = "PK1234567890ABCDEF"
    os.environ["ALPACA_SECRET_KEY"] = "SK1234567890ABCDEFGHIKLMNOPQRSTUVWXY"

    # Fetch 15-minute data for AAPL for the last 7 days
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    print(f"Fetching AAPL 15-minute data from {from_date} to {to_date}...")
    aapl_data = fetcher.get_candlestick_data("AAPL", 15, "minute", from_date, to_date)
    print(aapl_data.head())

    # Fetch 1-day data for MSFT for the last 30 days
    print("\nFetching MSFT 1-day data for last 30 days...")
    to_date_d = datetime.now().strftime('%Y-%m-%d')
    from_date_d = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    msft_data = fetcher.get_candlestick_data("MSFT", 1, "day", from_date_d, to_date_d)
    print(msft_data.head())

