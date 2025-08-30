
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';
import { BarChart3, TrendingUp, TrendingDown, Volume2, Settings, Play, Pause, ChevronsRight } from 'lucide-react';

const TradingChart = () => {
    const [selectedSymbol, setSelectedSymbol] = useState('SPY');
    const [timeframe, setTimeframe] = useState('15m');
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentDataPoint, setCurrentDataPoint] = useState(0);

    const chartContainerRef = useRef();
    const chartRef = useRef();
    const candlestickSeriesRef = useRef();
    const volumeSeriesRef = useRef();

    // Generate realistic candlestick data
    const generateCandlestickData = (symbol, count = 200) => {
        const data = [];
        let basePrice = symbol === 'SPY' ? 640 : symbol === 'AAPL' ? 180 : symbol === 'TSLA' ? 250 : 50;
        for (let i = 0; i < count; i++) {
            const volatility = 0.03;
            const trend = Math.sin(i * 0.05) * 0.002;
            const open = i === 0 ? basePrice : data[i - 1].close;
            const change = (Math.random() - 0.5) * volatility + trend;
            const close = open * (1 + change);
            const high = Math.max(open, close) * (1 + Math.random() * 0.015);
            const low = Math.min(open, close) * (1 - Math.random() * 0.015);
            const volume = Math.random() * 1500000 + 700000;
            data.push({
                time: (Date.now() / 1000) - (count - i) * 15 * 60,
                open: parseFloat(open.toFixed(2)),
                high: parseFloat(high.toFixed(2)),
                low: parseFloat(low.toFixed(2)),
                close: parseFloat(close.toFixed(2)),
                volume: Math.floor(volume),
                color: close >= open ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)',
            });
        }
        return data;
    };

    const [chartData, setChartData] = useState(() => generateCandlestickData(selectedSymbol));

    const currentCandle = useMemo(() => chartData[currentDataPoint], [chartData, currentDataPoint]);
    const previousCandle = useMemo(() => chartData[Math.max(0, currentDataPoint - 1)], [chartData, currentDataPoint]);
    const priceChange = useMemo(() => currentCandle ? currentCandle.close - previousCandle.close : 0, [currentCandle, previousCandle]);
    const priceChangePercent = useMemo(() => previousCandle ? (priceChange / previousCandle.close) * 100 : 0, [priceChange, previousCandle]);

    useEffect(() => {
        const newData = generateCandlestickData(selectedSymbol);
        setChartData(newData);
        setCurrentDataPoint(newData.length - 1);
    }, [selectedSymbol]);

    useEffect(() => {
        if (chartContainerRef.current) {
            chartRef.current = createChart(chartContainerRef.current, {
                width: chartContainerRef.current.clientWidth,
                height: 500,
                layout: {
                    backgroundColor: '#131722',
                    textColor: 'rgba(255, 255, 255, 0.9)',
                },
                grid: {
                    vertLines: { color: '#2A2E39' },
                    horzLines: { color: '#2A2E39' },
                },
                crosshair: {
                    mode: CrosshairMode.Normal,
                },
                rightPriceScale: {
                    borderColor: '#485158',
                },
                timeScale: {
                    borderColor: '#485158',
                },
            });

            candlestickSeriesRef.current = chartRef.current.addCandlestickSeries({
                upColor: '#10B981',
                downColor: '#EF4444',
                borderDownColor: '#EF4444',
                borderUpColor: '#10B981',
                wickDownColor: '#EF4444',
                wickUpColor: '#10B981',
            });

            volumeSeriesRef.current = chartRef.current.addHistogramSeries({
                color: '#26a69a',
                priceFormat: {
                    type: 'volume',
                },
                priceScaleId: 'volume_scale',
                scaleMargins: {
                    top: 0.8,
                    bottom: 0,
                },
            });
            chartRef.current.priceScale('volume_scale').applyOptions({
                scaleMargins: {
                    top: 0.8,
                    bottom: 0,
                },
            });


            const handleResize = () => {
                chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
            };

            window.addEventListener('resize', handleResize);
            return () => {
                window.removeEventListener('resize', handleResize);
                chartRef.current.remove();
            };
        }
    }, [chartContainerRef]);

    useEffect(() => {
        const visibleData = chartData.slice(0, currentDataPoint + 1);
        candlestickSeriesRef.current.setData(visibleData);
        const volumeData = visibleData.map(d => ({ time: d.time, value: d.volume, color: d.color }));
        volumeSeriesRef.current.setData(volumeData);
        chartRef.current.timeScale().setVisibleRange({
            from: chartData[Math.max(0, currentDataPoint - 50)].time,
            to: chartData[currentDataPoint].time,
        });
    }, [chartData, currentDataPoint]);


    useEffect(() => {
        if (!isPlaying) return;
        const interval = setInterval(() => {
            setCurrentDataPoint(prev => {
                if (prev >= chartData.length - 1) {
                    setIsPlaying(false);
                    return prev;
                }
                return prev + 1;
            });
        }, 200);
        return () => clearInterval(interval);
    }, [isPlaying, chartData.length]);

    return (
        <div className="bg-gray-900 text-white p-4 font-sans antialiased">
            {/* Header */}
            <div className="flex items-center justify-between mb-4 bg-gray-800 p-3 rounded-lg shadow-md">
                <div className="flex items-center space-x-4">
                    <select value={selectedSymbol} onChange={(e) => setSelectedSymbol(e.target.value)} className="bg-gray-700 text-white px-3 py-1.5 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500" >
                        <option value="SPY">SPY</option>
                        <option value="AAPL">AAPL</option>
                        <option value="TSLA">TSLA</option>
                        <option value="MSFT">MSFT</option>
                    </select>
                    <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)} className="bg-gray-700 text-white px-3 py-1.5 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500" >
                        <option value="1m">1m</option>
                        <option value="5m">5m</option>
                        <option value="15m">15m</option>
                        <option value="1h">1h</option>
                        <option value="1d">1d</option>
                    </select>
                    <button onClick={() => setIsPlaying(!isPlaying)} className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 px-4 py-1.5 rounded-md transition-colors" >
                        {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                        <span className="font-semibold">{isPlaying ? 'Pause' : 'Play'}</span>
                    </button>
                </div>
                <div className="flex items-center space-x-6">
                    <button className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors">
                        <BarChart3 size={18} />
                        <span>Indicators</span>
                    </button>
                    <button className="text-gray-300 hover:text-white transition-colors">
                        <Settings size={20} />
                    </button>
                </div>
            </div>

            {/* Price Info */}
            <div className="mb-4 bg-gray-800 p-4 rounded-lg shadow-md">
                <div className="flex items-center space-x-8">
                    <div>
                        <span className="text-gray-400 text-sm">Symbol</span>
                        <p className="text-2xl font-bold">{selectedSymbol}</p>
                    </div>
                    <div>
                        <span className="text-gray-400 text-sm">Price</span>
                        <p className="text-2xl font-bold">${currentCandle?.close.toFixed(2)}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                        {priceChange >= 0 ? <TrendingUp size={20} className="text-green-500" /> : <TrendingDown size={20} className="text-red-500" />}
                        <div>
                            <span className="text-gray-400 text-sm">Change</span>
                            <p className={`font-semibold ${priceChange >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} ({priceChangePercent.toFixed(2)}%)
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center space-x-2">
                        <Volume2 size={20} className="text-gray-400" />
                        <div>
                            <span className="text-gray-400 text-sm">Volume</span>
                            <p className="font-semibold">{(currentCandle?.volume || 0).toLocaleString()}</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Chart */}
            <div className="bg-gray-800 rounded-lg shadow-md" ref={chartContainerRef} style={{ height: '500px' }}></div>

            {/* Controls */}
            <div className="mt-4 flex items-center justify-between bg-gray-800 p-3 rounded-lg shadow-md">
                <div className="flex items-center space-x-4 text-sm text-gray-400">
                    <span>Data Points: {chartData.length}</span>
                    <span>Current: {currentDataPoint + 1}</span>
                </div>
                <div className="flex items-center space-x-3">
                    <input
                        type="range"
                        min="50"
                        max={chartData.length - 1}
                        value={currentDataPoint}
                        onChange={(e) => setCurrentDataPoint(parseInt(e.target.value))}
                        className="w-48 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <button onClick={() => setCurrentDataPoint(chartData.length - 1)} className="flex items-center space-x-1 bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded-md text-sm font-semibold transition-colors" >
                        <span>Latest</span>
                        <ChevronsRight size={16} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default TradingChart;
