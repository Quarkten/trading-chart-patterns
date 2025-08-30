import React, { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';

const MinimalChart = () => {
    const chartContainerRef = useRef();

    useEffect(() => {
        if (!chartContainerRef.current) {
            return;
        }

        console.log("Creating minimal chart...");
        const chart = createChart(chartContainerRef.current, {
            width: 600,
            height: 400,
            layout: {
                backgroundColor: '#131722',
                textColor: 'rgba(255, 255, 255, 0.9)',
            },
            grid: {
                vertLines: { color: '#2A2E39' },
                horzLines: { color: '#2A2E39' },
            },
        });

        const lineSeries = chart.addLineSeries();

        const data = [
            { time: '2019-04-11', value: 80.01 },
            { time: '2019-04-12', value: 96.63 },
            { time: '2019-04-13', value: 76.64 },
            { time: '2019-04-14', value: 81.89 },
            { time: '2019-04-15', value: 74.43 },
            { time: '2019-04-16', value: 80.01 },
            { time: '2019-04-17', value: 96.63 },
            { time: '2019-04-18', value: 76.64 },
            { time: '2019-04-19', value: 81.89 },
            { time: '2019-04-20', value: 74.43 },
        ];

        lineSeries.setData(data);

        return () => {
            console.log("Cleaning up minimal chart...");
            chart.remove();
        };
    }, []);

    return <div ref={chartContainerRef} style={{ width: '600px', height: '400px' }} />;
};

export default MinimalChart;
