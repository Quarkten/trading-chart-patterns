import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.data_fetcher import DataFetcher
from alpaca.data.live import StockDataStream
import websockets
import json

# Mock environment variables
@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict('os.environ', {
        'ALPACA_API_KEY': 'test_alpaca_key',
        'ALPACA_SECRET_KEY': 'test_alpaca_secret',
        'FINNHUB_API_KEY': 'test_finnhub_key'
    }):
        yield

# Fixture for DataFetcher instance
@pytest.fixture
def data_fetcher_instance():
    mock_socketio = MagicMock()
    fetcher = DataFetcher(socketio=mock_socketio)
    return fetcher

@pytest.mark.asyncio
async def test_alpaca_reconnect_on_error(data_fetcher_instance):
    fetcher = data_fetcher_instance
    
    with patch('src.data_fetcher.StockDataStream') as MockStockDataStream:
        # Configure the side_effect for MockStockDataStream itself
        # Each call to StockDataStream() will return a new mock client
        MockStockDataStream.side_effect = [
            MagicMock(run=AsyncMock(side_effect=Exception("Simulated Alpaca error"))),
            MagicMock(run=AsyncMock(side_effect=Exception("Another simulated Alpaca error"))),
            MagicMock(run=AsyncMock(side_effect=asyncio.CancelledError)) # Simulate a long-running task that gets cancelled
        ]

        symbols = ["SPY"]
        
        stream_task = asyncio.create_task(fetcher.start_realtime_stream(symbols))
        await asyncio.sleep(3.5) # Allow enough time for all retries to occur
        fetcher.should_run = False # Signal the stream to stop
        await stream_task # Wait for the stream task to complete its shutdown

        # Now, assert on the call count of MockStockDataStream itself
        assert MockStockDataStream.call_count == 3
        # We can't easily assert on the run() method of each individual mock client
        # unless we capture them. For now, asserting the instantiation count is sufficient.

@pytest.mark.asyncio
async def test_alpaca_reconnect_on_connection_limit(data_fetcher_instance):
    fetcher = data_fetcher_instance
    
    with patch('src.data_fetcher.StockDataStream') as MockStockDataStream:
        # Simulate "connection limit exceeded" errors for MAX_ALPACA_RETRIES times, then success
        side_effects = [
            ValueError("connection limit exceeded") for _ in range(fetcher.MAX_ALPACA_RETRIES - 1)
        ]
        side_effects.append(AsyncMock(side_effect=asyncio.CancelledError)) # Success on the last attempt

        MockStockDataStream.side_effect = [
            MagicMock(run=AsyncMock(side_effect=se)) for se in side_effects
        ]

        symbols = ["SPY"]
        
        stream_task = asyncio.create_task(fetcher.start_realtime_stream(symbols))
        # Calculate total sleep time needed for all retries
        total_sleep_time = sum([1 * (2 ** i) for i in range(fetcher.MAX_ALPACA_RETRIES - 1)]) + 0.5 # Add a small buffer
        await asyncio.sleep(total_sleep_time)
        fetcher.should_run = False
        await stream_task

        assert MockStockDataStream.call_count == fetcher.MAX_ALPACA_RETRIES

@pytest.mark.asyncio
async def test_finnhub_reconnect_on_error(data_fetcher_instance):
    fetcher = data_fetcher_instance
    
    with patch('src.data_fetcher.websockets.connect') as mock_finnhub_connect:
        mock_websocket = AsyncMock()
        mock_finnhub_connect.return_value.__aenter__.return_value = mock_websocket
        
        # Simulate a connection error after one message, then a successful connection
        mock_websocket.recv.side_effect = [
            json.dumps({"type": "trade", "data": [{"s": "SPY", "p": 100.0, "t": 1678886400000, "v": 1000}]}),
            websockets.exceptions.ConnectionClosedError(rcvd=1000, sent=1000, rcvd_then_sent=False), # Simulate an error
            json.dumps({"type": "trade", "data": [{"s": "SPY", "p": 101.0, "t": 1678886401000, "v": 1001}]}),
            websockets.exceptions.ConnectionClosedOK(rcvd=1000, sent=1000, rcvd_then_sent=True) # Graceful shutdown
        ]

        symbols = ["SPY"]
        
        # Start the Finnhub stream directly (as if Alpaca fallback occurred)
        stream_task = asyncio.create_task(fetcher._finnhub_websocket_connect(symbols))
        await asyncio.sleep(6) # Allow time for error and reconnection attempt (5s sleep + buffer)
        fetcher.should_run = False # Signal the stream to stop
        await stream_task

        # Verify that connect was called multiple times (initial + 1 reconnect)
        assert mock_finnhub_connect.call_count >= 2
        mock_websocket.send.assert_called_with(json.dumps({"type": "subscribe", "symbol": "SPY"}))
        assert fetcher.socketio.emit.call_count == 2 # Two messages should have been emitted