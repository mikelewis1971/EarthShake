import sys
# Install necessary libraries if they aren't already present
# !{sys.executable} -m pip install obspy scikit-learn matplotlib scipy

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from scipy.ndimage import gaussian_filter1d # New import for Gaussian filter

class SeismicCandleAnalyzer:
    """
    A class to fetch seismic waveform data, convert it into candle-like structures,
    apply various financial technical indicators, and visualize/save the results.
    """

    def __init__(self, network, station, location, channel, start_time_str, end_time_str):
        """
        Initializes the SeismicCandleAnalyzer with FDSN client and station parameters.

        Args:
            network (str): The network code (e.g., 'CU').
            station (str): The station code (e.g., 'GTBY').
            location (str): The location code (e.g., '00').
            channel (str): The channel code (e.g., 'BHZ').
            start_time_str (str): Start date and time string (e.g., '29/01/2020 00:00:00').
            end_time_str (str): End date and time string (e.g., '29/01/2020 03:00:00').
        """
        self.client = Client("IRIS")
        self.network = network
        self.station = station
        self.location = location
        self.channel = channel
        self.start_time_str = start_time_str
        self.end_time_str = end_time_str
        self.trace = None  # To store the raw obspy Trace object
        self.df = None     # To store the raw waveform data as a DataFrame
        self.candles = {}  # To store generated candles for different intervals
        print(f"SeismicCandleAnalyzer initialized for {network}.{station}.{location}.{channel}.")
        print(f"Time range: {start_time_str} to {end_time_str}.")

    def _convert_time(self, date_time_string):
        """
        Converts a date/time string to an ObsPy UTCDateTime object.

        Args:
            date_time_string (str): Date and time string in '%d/%m/%Y %H:%M:%S' format.

        Returns:
            UTCDateTime: The converted UTCDateTime object.
        """
        date_object = datetime.strptime(date_time_string, "%d/%m/%Y %H:%M:%S")
        return UTCDateTime(date_object)

    def fetch_data(self):
        """
        Fetches seismic waveform data from the FDSN client and stores it as a Pandas DataFrame.
        Stores the raw trace in self.trace and DataFrame in self.df.
        """
        print("
--- Fetching Waveform Data ---")
        start_time = self._convert_time(self.start_time_str)
        end_time = self._convert_time(self.end_time_str)

        print(f"  Requesting data for {self.network}.{self.station}.{self.location}.{self.channel} from {start_time} to {end_time}...")
        try:
            waveform = self.client.get_waveforms(self.network, self.station, self.location, self.channel, start_time, end_time)
            if not waveform: # Check if waveform stream is empty
                raise ValueError("No waveform data returned for the specified parameters and time range.")

            self.trace = waveform[0]  # Select the first Trace from the Stream

            # Create a DataFrame from the trace data
            data = {
                "Timestamp": pd.to_datetime([self.trace.stats.starttime.datetime + timedelta(seconds=t) for t in self.trace.times()]),
                "Amplitude": self.trace.data
            }
            self.df = pd.DataFrame(data)
            print(f"  Successfully fetched {len(self.df)} data points and stored in self.df.")
            print(f"  Data start: {self.df['Timestamp'].min()}, Data end: {self.df['Timestamp'].max()}")
        except Exception as e:
            print(f"  Error fetching data: {e}")
            self.trace = None
            self.df = None

    def _create_candles(self, minutes):
        """
        Generates candle-like data (OHLC-like, but with average, min, max, first, last amplitude)
        from the raw waveform data, applying MinMaxScaler for normalization.

        Args:
            minutes (int): The duration of each candle in minutes.

        Returns:
            pd.DataFrame: A DataFrame containing the generated candles.
        """
        if self.df is None or self.trace is None:
            print("  Error: No waveform data available to create candles. Please run fetch_data first.")
            return pd.DataFrame()

        print(f"  Creating {minutes}-minute candles...")
        # Calculate samples per minute and per candle
        # trace.stats.delta is the sampling interval in seconds
        samples_per_minute = int(60 / (self.trace.stats.delta))
        samples_per_candle = samples_per_minute * minutes

        # Ensure we have enough data for at least one candle
        if len(self.df) < samples_per_candle:
            print(f"  Warning: Not enough data ({len(self.df)} samples) to create a single {minutes}-minute candle (requires {samples_per_candle} samples).")
            return pd.DataFrame()

        num_candles = len(self.df) // samples_per_candle

        # Initialize DataFrame for candles
        candles_list = []

        scaler = MinMaxScaler() # Initialize MinMaxScaler for normalization

        for i in range(num_candles):
            candle_data = self.df.iloc[i * samples_per_candle:(i + 1) * samples_per_candle]

            start_time = candle_data["Timestamp"].iloc[0]
            end_time = candle_data["Timestamp"].iloc[-1]

            # Apply min-max scaling to the Amplitude column for the current candle data
            amplitude_values = candle_data["Amplitude"].values.reshape(-1, 1) # Reshape for scaler
            scaled_amplitude = scaler.fit_transform(amplitude_values)

            # Extract OHLC-like values from scaled amplitude
            low_val = scaled_amplitude.min()
            high_val = scaled_amplitude.max()
            first_val = scaled_amplitude[0][0]
            last_val = scaled_amplitude[-1][0]
            avg_val = (first_val + last_val + low_val + high_val) / 4 # Simple average of OHLC-like values

            candles_list.append({"Start Time": start_time, "End Time": end_time,
                                 "Low": low_val, "High": high_val, "First": first_val,
                                 "Last": last_val, "Average": avg_val})

        candles_df = pd.DataFrame(candles_list)
        print(f"  Successfully created {len(candles_df)} {minutes}-minute candles.")
        return candles_df

    def _calculate_ma(self, series, window):
        """
        Calculates the Simple Moving Average (MA) for a given pandas Series.
        Args:
            series (pd.Series): The input data series (e.g., 'Average' amplitude).
            window (int): The number of periods for the MA calculation.
        Returns:
            pd.Series: The Simple Moving Average series.
        """
        print(f"    Calculating Simple Moving Average (MA) with window={window}...")
        return series.rolling(window=window).mean()

    def _calculate_ema(self, series, window):
        """
        Calculates the Exponential Moving Average (EMA) for a given pandas Series.
        Args:
            series (pd.Series): The input data series (e.g., 'Average' amplitude).
            window (int): The number of periods for the EMA calculation.
        Returns:
            pd.Series: The Exponential Moving Average series.
        """
        print(f"    Calculating Exponential Moving Average (EMA) with span={window}...")
        return series.ewm(span=window, adjust=False).mean()

    def _calculate_bollinger_bands(self, series, window, num_std_dev):
        """
        Calculates Bollinger Bands (Upper, Middle, Lower) for a given pandas Series.
        Args:
            series (pd.Series): The input data series (e.g., 'Average' amplitude).
            window (int): The number of periods for the MA and Std Dev calculation.
            num_std_dev (int): The number of standard deviations for the upper/lower bands.
        Returns:
            tuple: A tuple containing (Middle Band (MA), Upper Band, Lower Band) series.
        """
        print(f"    Calculating Bollinger Bands (BB) with window={window}, num_std_dev={num_std_dev}...")
        ma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper_band = ma + (std * num_std_dev)
        lower_band = ma - (std * num_std_dev)
        return ma, upper_band, lower_band

    def _calculate_macd(self, series, fast_window, slow_window, signal_window):
        """
        Calculates Moving Average Convergence Divergence (MACD) and related components.
        Args:
            series (pd.Series): The input data series (e.g., 'Average' amplitude).
            fast_window (int): The window for the fast EMA.
            slow_window (int): The window for the slow EMA.
            signal_window (int): The window for the signal line EMA.
        Returns:
            tuple: MACD Line, Signal Line, MACD Histogram, Buy Signals (boolean series), Sell Signals (boolean series).
        """
        print(f"    Calculating MACD with fast_window={fast_window}, slow_window={slow_window}, signal_window={signal_window}...")
        exp1 = series.ewm(span=fast_window, adjust=False).mean()
        exp2 = series.ewm(span=slow_window, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        macd_histogram = macd_line - signal_line

        # Detect crossover points for buy/sell signals
        # Buy signal: MACD line crosses above Signal line
        buy_signals = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line > signal_line)
        # Sell signal: MACD line crosses below Signal line
        sell_signals = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line < signal_line)

        return macd_line, signal_line, macd_histogram, buy_signals, sell_signals

    def _calculate_rsi(self, series, window):
        """
        Calculates the Relative Strength Index (RSI) for a given pandas Series.
        Args:
            series (pd.Series): The input data series (e.g., 'Average' amplitude).
            window (int): The number of periods for RSI calculation.
        Returns:
            pd.Series: The RSI series.
        """
        print(f"    Calculating Relative Strength Index (RSI) with window={window}...")
        delta = series.diff() # Calculate price change
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean() # Average gains
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean() # Average losses

        # Handle division by zero for rs where loss is 0 to avoid warnings
        rs = np.where(loss == 0, np.inf, gain / loss)
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(rsi, index=series.index)

    def _calculate_stochastic_oscillator(self, high, low, close, k_window, d_window):
        """
        Calculates the %K and %D lines for the Stochastic Oscillator.
        Args:
            high (pd.Series): High values for each candle.
            low (pd.Series): Low values for each candle.
            close (pd.Series): Closing (Average) values for each candle.
            k_window (int): The number of periods for %K calculation.
            d_window (int): The number of periods for %D (SMA of %K) calculation.
        Returns:
            tuple: %K line series, %D line series.
        """
        print(f"    Calculating Stochastic Oscillator (%K, %D) with K_window={k_window}, D_window={d_window}...")
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()

        # Handle division by zero when highest_high == lowest_low
        percent_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        percent_k = percent_k.replace([np.inf, -np.inf], np.nan).fillna(0) # Replace inf with NaN and then 0 if no range

        percent_d = percent_k.rolling(window=d_window).mean()
        return percent_k, percent_d

    def _apply_gaussian_filter(self, series, sigma):
        """
        Applies a Gaussian filter to a given pandas Series.
        Args:
            series (pd.Series): The input data series (e.g., 'Average' amplitude).
            sigma (float): The standard deviation for the Gaussian kernel.
        Returns:
            pd.Series: The Gaussian filtered series.
        """
        print(f"    Applying Gaussian Filter with sigma={sigma}...")
        # Gaussian filter requires non-NaN values. Ffill to handle initial NaNs from rolling calculations.
        # Updated fillna(method='ffill') to ffill() to address FutureWarning
        return pd.Series(gaussian_filter1d(series.ffill().values, sigma=sigma), index=series.index)

    def plot_candles(self, candles_df, title, plot_indicators=False):
        """
        Enhanced plotting function to visualize candles and optional technical indicators.
        Uses subplots for clarity for certain indicators.

        Args:
            candles_df (pd.DataFrame): The DataFrame containing candle data and potentially indicators.
            title (str): The title for the plot.
            plot_indicators (bool): If True, plots additional technical indicators.
        """
        if candles_df.empty:
            print(f"No candles to plot for '{title}'.")
            return

        print(f"  Generating plot for '{title}'...")

        # Determine number of subplots needed
        if plot_indicators:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
            fig.suptitle(title, fontsize=16)
        else:
            fig, ax1 = plt.subplots(figsize=(15, 7))
            fig.suptitle(title, fontsize=16)
            ax2, ax3, ax4 = None, None, None # Initialize to None if not plotting indicators

        # Plot candles (Average) on the main axis (ax1)
        ax1.plot(candles_df['Start Time'], candles_df['Average'], label='Average Amplitude', linewidth=1.5, color='blue')

        if plot_indicators:
            # Plot MA and EMA if available
            if 'MA_50' in candles_df.columns:
                ax1.plot(candles_df['Start Time'], candles_df['MA_50'], label='MA (50)', color='red', linestyle='--')
            if 'EMA_20' in candles_df.columns:
                ax1.plot(candles_df['Start Time'], candles_df['EMA_20'], label='EMA (20)', color='green', linestyle='--')

            # Plot Bollinger Bands if available
            if 'BB_Upper' in candles_df.columns:
                ax1.plot(candles_df['Start Time'], candles_df['BB_Upper'], label='BB Upper', color='orange', linestyle=':')
                ax1.plot(candles_df['Start Time'], candles_df['BB_Middle'], label='BB Middle', color='gray', linestyle=':')
                ax1.plot(candles_df['Start Time'], candles_df['BB_Lower'], label='BB Lower', color='orange', linestyle=':')
                ax1.fill_between(candles_df['Start Time'], candles_df['BB_Lower'], candles_df['BB_Upper'], color='orange', alpha=0.1, label='Bollinger Band Range')

            # Plot Gaussian Filter if available
            if 'Gaussian_Filtered' in candles_df.columns:
                ax1.plot(candles_df['Start Time'], candles_df['Gaussian_Filtered'], label='Gaussian Filtered (sigma=5)', color='purple', linestyle='-.')

        ax1.set_ylabel('Scaled Amplitude')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        if plot_indicators:
            # Plot MACD on ax2
            if ax2 is not None and 'MACD_Line' in candles_df.columns:
                ax2.plot(candles_df['Start Time'], candles_df['MACD_Line'], label='MACD Line', color='blue')
                ax2.plot(candles_df['Start Time'], candles_df['Signal_Line'], label='Signal Line', color='red')
                ax2.bar(candles_df['Start Time'], candles_df['MACD_Histogram'], label='MACD Histogram', color='gray', alpha=0.7)
                # Plot crossover points
                buy_points = candles_df[candles_df['MACD_Buy_Signal']]
                sell_points = candles_df[candles_df['MACD_Sell_Signal']]
                if not buy_points.empty:
                    ax2.scatter(buy_points['Start Time'], buy_points['MACD_Line'], marker='^', color='green', s=100, label='Buy Signal', zorder=5)
                if not sell_points.empty:
                    ax2.scatter(sell_points['Start Time'], sell_points['MACD_Line'], marker='v', color='red', s=100, label='Sell Signal', zorder=5)

                ax2.set_ylabel('MACD')
                ax2.legend(loc='upper left')
                ax2.grid(True)

            # Plot RSI on ax3
            if ax3 is not None and 'RSI' in candles_df.columns:
                ax3.plot(candles_df['Start Time'], candles_df['RSI'], label='RSI', color='purple')
                ax3.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
                ax3.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
                ax3.set_ylabel('RSI')
                ax3.legend(loc='upper left')
                ax3.grid(True)
                ax3.set_ylim(0, 100)  # RSI usually ranges from 0 to 100

            # Plot Stochastic Oscillator on ax4
            if ax4 is not None and 'Stoch_K' in candles_df.columns:
                ax4.plot(candles_df['Start Time'], candles_df['Stoch_K'], label='%K Line', color='blue')
                ax4.plot(candles_df['Start Time'], candles_df['Stoch_D'], label='%D Line', color='red')
                ax4.axhline(80, color='red', linestyle='--', alpha=0.7, label='Overbought (80)')
                ax4.axhline(20, color='green', linestyle='--', alpha=0.7, label='Oversold (20)')
                ax4.set_xlabel('Time (Day of Month, Hour:Minute)')
                ax4.set_ylabel('Stochastic Oscillator')
                ax4.legend(loc='upper left')
                ax4.grid(True)
                ax4.set_ylim(0, 100)  # Stochastic Oscillator usually ranges from 0 to 100

            # Set x-axis formatter for all subplots and rotate labels for readability
            # Only the bottom-most plot needs x-axis labels visible
            plt.setp(ax1.get_xticklabels(), visible=False)
            if ax2 is not None: plt.setp(ax2.get_xticklabels(), visible=False)
            if ax3 is not None: plt.setp(ax3.get_xticklabels(), visible=False)
            if ax4 is not None:
                ax4.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

            plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent suptitle overlap
        else:
            # For plots without indicators, just format the single ax1
            ax1.set_xlabel('Time (Day of Month, Hour:Minute)')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()

        plt.show()
        print(f"  Plot for '{title}' displayed.")

    def save_candles_to_csv(self, candles_df, filename):
        """
        Saves a DataFrame of candles to a CSV file.

        Args:
            candles_df (pd.DataFrame): The DataFrame containing the candle data.
            filename (str): The name of the CSV file to save.
        """
        if candles_df.empty:
            print(f"  No candles to save to '{filename}'. CSV file will not be created.")
            return
        try:
            candles_df.to_csv(filename, index=False)
            print(f"  Candles saved to '{filename}'.")
        except Exception as e:
            print(f"  Error saving candles to CSV '{filename}': {e}")

    def run_analysis(self):
        """
        Orchestrates the entire analysis workflow: fetches data, creates candles,
        applies indicators, plots results, and saves data.
        """
        print("
--- Starting Seismic Candle Analysis Workflow ---")
        self.fetch_data()
        if self.df is None:
            print("Workflow terminated: Data fetching failed.")
            return

        print("
--- Generating Candles ---")
        # Generate candles for different time intervals and store them
        self.candles['1min'] = self._create_candles(1)
        self.candles['5min'] = self._create_candles(5)
        self.candles['15min'] = self._create_candles(15)
        self.candles['1hr'] = self._create_candles(60)

        # Apply indicators to a chosen candle DataFrame, e.g., 1-hour candles
        print("
--- Applying Financial Indicators to 1-hour Candles ---")
        candles_1hr = self.candles.get('1hr')
        if candles_1hr is not None and not candles_1hr.empty:
            # Ensure 'High', 'Low', 'Average' are numeric, handling potential issues
            candles_1hr[['High', 'Low', 'Average']] = candles_1hr[['High', 'Low', 'Average']].apply(pd.to_numeric, errors='coerce')
            candles_1hr.dropna(subset=['High', 'Low', 'Average'], inplace=True)
            if candles_1hr.empty:
                print("  Warning: 1-hour candles became empty after dropping NaNs for indicator calculation. Skipping indicators.")
            else:
                candles_1hr['MA_50'] = self._calculate_ma(candles_1hr['Average'], window=50)
                candles_1hr['EMA_20'] = self._calculate_ema(candles_1hr['Average'], window=20)

                candles_1hr['BB_Middle'], candles_1hr['BB_Upper'], candles_1hr['BB_Lower'] =                     self._calculate_bollinger_bands(candles_1hr['Average'], window=20, num_std_dev=2)

                candles_1hr['MACD_Line'], candles_1hr['Signal_Line'], candles_1hr['MACD_Histogram'],                 candles_1hr['MACD_Buy_Signal'], candles_1hr['MACD_Sell_Signal'] =                     self._calculate_macd(candles_1hr['Average'], fast_window=12, slow_window=26, signal_window=9)

                candles_1hr['RSI'] = self._calculate_rsi(candles_1hr['Average'], window=14)

                candles_1hr['Stoch_K'], candles_1hr['Stoch_D'] =                     self._calculate_stochastic_oscillator(candles_1hr['High'], candles_1hr['Low'], candles_1hr['Average'], k_window=14, d_window=3)

                candles_1hr['Gaussian_Filtered'] = self._apply_gaussian_filter(candles_1hr['Average'], sigma=5)
                print("  Indicators applied successfully to 1-hour candles.")

        else:
            print("  1-hour candles DataFrame is empty or None, skipping indicator calculations.")

        print("
--- Plotting Candles and Indicators ---")
        # Plot candles without indicators for shorter intervals
        self.plot_candles(self.candles.get('1min', pd.DataFrame()), '1-minute candles (Average)')
        self.plot_candles(self.candles.get('5min', pd.DataFrame()), '5-minute candles (Average)')
        self.plot_candles(self.candles.get('15min', pd.DataFrame()), '15-minute candles (Average)')
        # Plot 1-hour candles with all calculated indicators
        self.plot_candles(self.candles.get('1hr', pd.DataFrame()), '1-hour candles with Indicators', plot_indicators=True)

        print("
--- Saving Candles to CSV Files ---")
        # Save candles to CSV
        self.save_candles_to_csv(self.candles.get('1min', pd.DataFrame()), 'candles_1min.csv')
        self.save_candles_to_csv(self.candles.get('5min', pd.DataFrame()), 'candles_5min.csv')
        self.save_candles_to_csv(self.candles.get('15min', pd.DataFrame()), 'candles_15min.csv')
        self.save_candles_to_csv(self.candles.get('1hr', pd.DataFrame()), 'candles_1hr_with_indicators.csv')
        print("  All candles saved to CSV files.")

        print("
--- Seismic Candle Analysis Workflow Complete ---")


# --- Demonstration Code ---
# This part of the code is for demonstration within a notebook or script.
# If you save this class to a .py file, you might want to put the demonstration
# code under an `if __name__ == '__main__':` block.

# print("
--- Initiating SeismicCandleAnalyzer Demonstration ---")

# Define the network, station, channel, and a smaller, more manageable time period
# network = "CU"
# station = "GTBY"
# location = '00'
# channel = "BHZ"
# start_time_str = "29/01/2020 00:00:00"
# end_time_str = "29/01/2020 03:00:00" # Adjusted to 3 hours for demonstration

# Instantiate the class with the new parameters
# analyzer = SeismicCandleAnalyzer(network, station, location, channel, start_time_str, end_time_str)

# Run the analysis workflow
# analyzer.run_analysis()

# print("
--- Demonstration Finished ---")
