
import requests
from requests.exceptions import HTTPError
import pandas as pd
import os
import logging
import numpy as np


def get_raw_frequency_data(year, month):
    file_path = f"raw_data/fnew-{year}-{month}.parquet"
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    
    csv_path = f"raw_data/fnew-{year}-{month}.csv"
    if not os.path.exists("raw_data"):
        os.makedirs("raw_data")
    
    try:
        df = pd.read_csv(csv_path)
        df.to_parquet(file_path, index=False)
        os.remove(csv_path)
        return df
    except Exception as e:
        logging.error(f"Error loading raw data for {year}-{month}: {e}")
        return None
    
def perform_fft_analysis(data_source):
   fs = 1  # 1 Hz sampling rate
   values = data_source['f'].values - np.mean(data_source['f'].values)
   
   # Apply Hanning window
   window = np.hanning(len(values))
   windowed_values = values * window
   
   fft_result = np.fft.fft(windowed_values)
   # Compensate for energy loss due to windowing (optional)
   fft_result = fft_result * (2.0 / np.sum(window))
   
   magnitudes = np.abs(fft_result[:len(fft_result)//2])
   # normalize magnitude
   freq_bins = np.fft.fftfreq(len(values), d=1/fs)[:len(values)//2]
   
   freq_dict = {}
   for freq, mag in zip(freq_bins, magnitudes):
        # Ignore the DC component
        if freq > 0.1:
            freq_dict[f"{freq:.4f}"] = mag
   

   return freq_dict

def get_national_grid_data():
    local_file = "df_fuel_ckan.csv"
    url = "https://api.neso.energy/dataset/88313ae5-94e4-4ddc-a790-593554d8c6b9/resource/f93d1835-75bc-43e5-84ad-12472b180a98/download/df_fuel_ckan.csv"
    
    if os.path.exists(local_file):
        logging.info(f"Loading data from local file: {local_file}")
        return pd.read_csv(local_file)
    
    logging.info(f"Downloading data from {url}")
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        
        with open(local_file, "wb") as file:
            file.write(response.content)
        
        return pd.read_csv(local_file)
    except HTTPError as e:
        logging.error(f"HTTP Error: {e}")
    except Exception as e:
        logging.error(f"Error: {e}")
    
    return None