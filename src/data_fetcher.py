import os
import asyncio
import json
import websockets
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import random

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical import StockHistoricalDataClient

load_dotenv() # Load environment variables from .env file

from alpaca.data.live import StockDataStream

class DataFetcher:
    MAX_ALPACA_RETRIES = 3 # Reduced retries for quicker fallback
    MAX_FINNHUB_RETRIES = 5 # Max retries for Finnhub websocket connection

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
        self.websocket_client = None
        self.stream_loop = None

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

        finnhub_retry_count = 0
        BASE_FINNHUB_DELAY = 5 # seconds
        while self.should_run and finnhub_retry_count < self.MAX_FINNHUB_RETRIES:
            try:
                print(f"Attempting to connect to Finnhub websocket (Attempt {finnhub_retry_count + 1}/{self.MAX_FINNHUB_RETRIES})...")
                async with websockets.connect(finnhub_ws_url) as websocket:
                    print(f"Successfully connected to Finnhub websocket for symbols: {symbols}")
                    # Reset retry count on successful connection
                    finnhub_retry_count = 0
                    # Subscribe to trades for each symbol
                    for symbol in symbols:
                        subscribe_message = json.dumps({"type": "subscribe", "symbol": symbol})
                        await websocket.send(subscribe_message)
                        print(f"Sent subscription message for {symbol}: {subscribe_message}")

                    while self.should_run:
                        message = await websocket.recv()
                        print(f"Received message from Finnhub: {message[:100]}...") # Log first 100 chars
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
                        else:
                            print(f"Received non-trade message from Finnhub: {data}")
            except websockets.exceptions.ConnectionClosedOK:
                print("Finnhub websocket connection closed gracefully.")
                break
            except websockets.exceptions.ConnectionClosedError as e:
                delay = BASE_FINNHUB_DELAY * (2 ** finnhub_retry_count) + random.uniform(0, 2)
                print(f"Finnhub websocket connection closed with error: {type(e).__name__}: {e}. Attempting to reconnect in {delay:.2f} seconds...")
                finnhub_retry_count += 1
                await asyncio.sleep(delay)
            except websockets.exceptions.WebSocketException as e:
                delay = BASE_FINNHUB_DELAY * (2 ** finnhub_retry_count) + random.uniform(0, 2)
                print(f"A Finnhub WebSocket-specific error occurred: {type(e).__name__}: {e}. Attempting to reconnect in {delay:.2f} seconds...")
                finnhub_retry_count += 1
                await asyncio.sleep(delay)
            except Exception as e:
                delay = BASE_FINNHUB_DELAY * (2 ** finnhub_retry_count) + random.uniform(0, 2)
                print(f"An unexpected error occurred with Finnhub websocket: {type(e).__name__}: {e}. Attempting to reconnect in {delay:.2f} seconds...")
                finnhub_retry_count += 1
                await asyncio.sleep(delay)
        
        if finnhub_retry_count >= self.MAX_FINNHUB_RETRIES:
            print(f"Max Finnhub retries ({self.MAX_FINNHUB_RETRIES}) reached. Stopping Finnhub real-time stream attempts.")

    async def start_realtime_stream(self, symbols: list):
        print(f"Starting real-time stream for symbols: {symbols}")
        self.should_run = True
        
        # Prioritize Finnhub for real-time data
        await self._finnhub_websocket_connect(symbols)

        # The Alpaca stream code is left here as a potential fallback,
        # but it will not be executed if the Finnhub stream runs.
        if self.should_run:
            print("Attempting to connect to Alpaca WebSocket client as a fallback...")
            try:
                self.websocket_client = StockDataStream(self.api_key, self.secret_key)
                self.websocket_client.subscribe_bars(self._alpaca_bar_data_handler, *symbols)
                await self.websocket_client.run()
                print("Alpaca WebSocket client stopped gracefully.")
            except Exception as e:
                print(f"Alpaca WebSocket stream failed: {type(e).__name__}: {e}.")
            finally:
                if self.websocket_client:
                    try:
                        await self.websocket_client.close()
                        print("Alpaca WebSocket client closed in finally block.")
                    except Exception as close_e:
                        print(f"Error closing Alpaca WebSocket client in finally block: {close_e}")

    async def stop_realtime_stream(self):
        print("Stopping real-time stream...")
        self.should_run = False # This will stop the Finnhub loop
        if self.websocket_client and self.websocket_client.is_running:
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

