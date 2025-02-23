import requests
import pandas as pd
import os
import logging
import numpy as np
from scipy import signal

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_raw_frequency_data(year, month):
    file_path = f"raw_data/fnew-{year}-{month}.parquet"
    if os.path.exists(file_path):
        logging.info(f"Raw data for {year}-{month} already exists")
        return
    logging.info(f"Downloading raw data for {year}-{month}")
    try:
        if not os.path.exists("raw_data"):
            os.makedirs("raw_data")
        
        #https://api.neso.energy/dataset/cb1cc925-ecd8-4406-b021-3a3f368196e1/resource/a1ccd82c-e522-4b5d-a4da-57dafab9d6de/download/fnew-2022-9.csv
        url = f"https://api.neso.energy/dataset/cb1cc925-ecd8-4406-b021-3a3f368196e1/resource/a1ccd82c-e522-4b5d-a4da-57dafab9d6de/download/fnew-{year}-{month}.csv"
        response = requests.get(url)
        with open(f"fnew-{year}-{month}.csv", "wb") as file:
            file.write(response.content)
        # write as parquet
        df = pd.read_csv(f"fnew-{year}-{month}.csv")
        df.to_parquet(file_path, index=False)
        os.remove(f"fnew-{year}-{month}.csv")
    except Exception as e:
        logging.error(f"Error downloading raw data for {year}-{month}: {e}")
        return None
    
def perform_fft_analysis(data_source, window_size=1024):
    """
    Performs FFT analysis on the full dataset
    
    Parameters:
    data_source: DataFrame with timestamp and value columns
    window_size: Size of the FFT window (higher = better frequency resolution)
    """
    
    # Get the data
    values = data_source['value'].values
    
    # Remove DC component (mean)
    values = values - np.mean(values)
    
    # Apply Hamming window
    windowed = values * signal.windows.hamming(len(values))
    
    # Perform FFT
    fft_result = np.fft.fft(windowed)
    
    # Convert to magnitude and take only first half (positive frequencies)
    magnitudes = np.abs(fft_result[:len(fft_result)//2])
    
    return magnitudes

def analyze_frequency_data(file_path):
    """
    Analyze frequency data from a parquet file
    """
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        # Ensure data is sorted by timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate sampling rate from timestamps
        sampling_rate = 1.0 / (df['timestamp'].diff().median().total_seconds())
        
        # Perform FFT analysis
        fft_magnitudes = perform_fft_analysis(df)
        
        # Calculate frequency bins
        freq_bins = np.fft.fftfreq(len(df), d=1/sampling_rate)[:len(df)//2]
        
        # Create results dictionary
        results = {
            'frequencies': freq_bins,
            'magnitudes': fft_magnitudes,
            'sampling_rate': sampling_rate,
            'max_magnitude_freq': freq_bins[np.argmax(fft_magnitudes)],
            'max_magnitude': np.max(fft_magnitudes)
        }
        
        return results
    
    except Exception as e:
        logging.error(f"Error analyzing frequency data: {e}")
        return None

if __name__ == "__main__":
    for date in pd.date_range("2014-01-01", "2025-02-01", freq="1MS"):
        get_raw_frequency_data(date.year, date.month)
