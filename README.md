
# Seismic Candle Analyzer: Unveiling Patterns in Earth's Rhythms

Welcome to the Seismic Candle Analyzer! This tool helps you explore and understand seismic (earthquake-related) data in a new and exciting way. Imagine looking at earthquake activity like you would stock market prices – using "candles" and special indicators to spot trends.

## What is this project about?

This project provides a user-friendly way to:
1.  **Fetch seismic data**: Automatically download raw measurements from earthquake sensors around the world.
2.  **Create "seismic candles"**: Transform the raw data into visual patterns, similar to stock market candlesticks, which highlight changes in seismic amplitude over time.
3.  **Apply financial indicators**: Use well-known financial analysis tools (like moving averages or Bollinger Bands) to identify potential trends, reversals, or unusual activity in the seismic data.
4.  **Visualize and save results**: Display these patterns and indicators on charts and save them for further study.

The ultimate goal is to explore if these indicators, traditionally used for financial markets, can help us anticipate earthquake activity.

## Understanding the Basics

### What is IRIS?

**IRIS** stands for *Incorporated Research Institutions for Seismology*. It's a consortium that operates a global network of seismic sensors and provides open access to seismic data for research and public education. When our tool "fetches data," it's connecting to the IRIS data center to get the raw seismic measurements.

### What do Network, Station, Location, and Channel mean?

Think of seismic data collection like weather stations. Each parameter helps us pinpoint *where* and *how* the data was collected:
*   **Network (e.g., 'CU')**: A group of seismic stations operated by a specific organization or country. 'CU' is the code for the China University of Geosciences network.
*   **Station (e.g., 'GTBY')**: A specific physical location where seismic sensors are installed. 'GTBY' is the code for a particular station within the 'CU' network.
*   **Location (e.g., '00')**: Sometimes a station has multiple sensors at slightly different positions. The location code distinguishes these. '00' often means the primary sensor or a default location.
*   **Channel (e.g., 'BHZ')**: Describes the type of sensor and what it measures:
    *   **BH**: Stands for "Broadband High Gain." These sensors are very sensitive and can detect a wide range of seismic frequencies.
    *   **Z**: Indicates the vertical component of ground motion. Other common channels might measure horizontal motion (N for North-South, E for East-West).

### What are "Seismic Candles"?

Inspired by financial candlesticks, seismic candles summarize seismic activity over a specific time interval (e.g., 1 minute, 5 minutes, 1 hour). Each candle represents:
*   **Start Time / End Time**: The beginning and end of the interval.
*   **Low / High**: The minimum and maximum scaled seismic amplitude recorded during that interval.
*   **First / Last**: The scaled seismic amplitude at the very beginning and end of the interval.
*   **Average**: A normalized average amplitude over the interval. These candles simplify vast amounts of raw data into understandable patterns.

## The Financial Indicators and What They Tell Us

These indicators, usually for market analysis, can help us look for trends and unusual activity in seismic data:

*   **Moving Average (MA)**: This is simply the average "seismic amplitude" over a certain number of past candles (e.g., last 50 candles). It helps to smooth out short-term fluctuations and identify the general trend. If the amplitude is generally above the MA, it might suggest increasing activity; below, decreasing activity.

*   **Exponential Moving Average (EMA)**: Similar to MA, but it gives more weight to recent data points. This makes it more responsive to new information and quicker to reflect changes in seismic activity.

*   **Bollinger Bands (BB)**: These consist of a middle band (usually an MA) and two outer bands (upper and lower) that represent a typical range of price (or seismic amplitude) movement. When the seismic amplitude goes outside these bands, it might indicate an unusual event or extreme activity.

*   **Moving Average Convergence Divergence (MACD)**: This indicator shows the relationship between two EMAs of different lengths. It helps spot changes in the strength, direction, momentum, and duration of a trend. A "crossover" (when two lines on the MACD chart cross) can sometimes signal a shift in seismic activity patterns, potentially indicating a "buy" (increasing activity) or "sell" (decreasing activity) signal in our earthquake anticipation context.

*   **Relative Strength Index (RSI)**: The RSI measures the speed and change of amplitude movements. It's an "oscillator" that moves between 0 and 100. High RSI values (e.g., above 70) might suggest "overbought" or excessively high seismic activity, while low values (e.g., below 30) might suggest "oversold" or unusually low activity.

*   **Stochastic Oscillator**: Similar to RSI, this indicator compares a particular seismic amplitude to its price range over a certain period. It aims to tell you if the seismic amplitude is near its high or low for that period. Values above 80 or below 20 could indicate unusual seismic conditions.

*   **Gaussian Filter**: This is a signal processing technique that smooths data by blurring out noise and fine-scale detail. Applied to seismic data, it can help reveal underlying patterns or trends that might be obscured by rapid fluctuations.

## How to Use the Seismic Candle Analyzer

1.  **Initialize the Analyzer**: Create an instance of the `SeismicCandleAnalyzer` class, providing the network, station, location, channel, and the start and end dates/times you want to analyze. For example:
    ```python
    analyzer = SeismicCandleAnalyzer(
        network="CU", 
        station="GTBY", 
        location='00', 
        channel="BHZ", 
        start_time_str="29/01/2020 00:00:00", 
        end_time_str="29/01/2020 03:00:00"
    )
    ```

2.  **Run the Analysis**: Simply call the `run_analysis()` method on your analyzer object:
    ```python
    analyzer.run_analysis()
    ```

That's it! The class will handle everything:
*   Fetching the raw data from IRIS.
*   Creating candles for different time intervals (1-minute, 5-minute, 15-minute, 1-hour).
*   Calculating all the financial indicators on the 1-hour candles.
*   Plotting the candles and indicators.
*   Saving all the candle data (including indicators) into CSV files in your current directory.

## What to Look For (and what it means for earthquake anticipation)

As you view the plots, especially the 1-hour candles with indicators, look for:
*   **Unusual Spikes or Dips**: Any sudden, large changes in the "Average Amplitude" that stand out from the normal pattern.
*   **Bollinger Band Breaches**: When the amplitude moves significantly outside the upper or lower Bollinger Bands, it could signal a period of high volatility or unusual energy release.
*   **RSI or Stochastic Extremes**: If RSI or Stochastic Oscillator values are consistently above 70-80 or below 20-30, it might indicate sustained periods of unusual seismic energy buildup or release.
*   **MACD Crossovers**: Shifts in MACD lines can indicate changes in the momentum of seismic activity. A strong upward cross might precede an increase in smaller tremors or a larger event, while a downward cross might suggest a decrease in activity.
*   **Trends in Moving Averages**: A consistent upward trend in MAs suggests increasing seismic activity over time, while a downward trend suggests the opposite.

**Important Note**: This tool is for *exploratory analysis* and *research purposes*. While financial indicators can reveal patterns, predicting earthquakes is a complex scientific challenge, and these methods are experimental. Always consult official seismic monitoring agencies for earthquake information and safety.
