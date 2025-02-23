import requests
from requests.exceptions import HTTPError
import pandas as pd
import os
import logging
import numpy as np
from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import shap

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
   fft_result = np.fft.fft(values)
   magnitudes = np.abs(fft_result[:len(fft_result)//2])
   freq_bins = np.fft.fftfreq(len(values), d=1/fs)[:len(values)//2]
   
   period_dict = {}
   for freq, mag in zip(freq_bins, magnitudes):
       if freq > 0 and freq < 0.1:  # 0-100mHz
           period_dict[f"period_{round(1/abs(freq), 3)}"] = mag
   
   return period_dict

def main():
    rows = []
    for date in pd.date_range("2024-06-01", "2025-01-01", freq="1MS"):
        df = get_raw_frequency_data(date.year, date.month)
        if df is None or df.empty:
            continue
        
        for i in range(0, len(df), 1800):
            try:
                timestamp = df.iloc[i]["dtm"]
                block = df.iloc[i:i+1800]
                fft_dict = perform_fft_analysis(block)
                fft_dict["timestamp"] = timestamp
                rows.append(fft_dict)
            except Exception as e:
                logging.error(f"Error processing block at index {i}: {e}")
    
    result_df = pd.DataFrame(rows)
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    result_df.set_index('timestamp', inplace=True)
    
    # Add rolling means
    fft_columns = [col for col in result_df.columns if col.startswith('period_')]
    for col in fft_columns:
        result_df[f'{col}_6h'] = result_df[col].rolling(window='6H').mean()
        result_df[f'{col}_24h'] = result_df[col].rolling(window='24H').mean()
    
    # # Add ratio features between FFT bands
    # for i, col1 in enumerate(fft_columns):
    #     for col2 in fft_columns[i+1:]:
    #         result_df[f'ratio_{col1}_{col2}'] = result_df[col1] / result_df[col2]
    
    fuel_data = get_national_grid_data()
    if fuel_data is None:
        logging.error("Failed to load fuel data. Exiting.")
        return
    
    fuel_data['DATETIME'] = pd.to_datetime(fuel_data['DATETIME'])
    fuel_data.set_index('DATETIME', inplace=True)
    fuel_data = fuel_data[['CARBON_INTENSITY']]
    
    result_df = pd.merge_asof(result_df, fuel_data, left_index=True, right_index=True)
    result_df.dropna(inplace=True)
    result_df.drop_duplicates(inplace=True)

    # Log transform carbon intensity
    result_df['CARBON_INTENSITY_LOG'] = np.log1p(result_df['CARBON_INTENSITY'])

    validation_df = result_df[result_df.index.year == 2025]
    result_df = result_df[result_df.index.year != 2025]
    
    # Weight higher carbon intensity samples more
    sample_weights = result_df['CARBON_INTENSITY'] / result_df['CARBON_INTENSITY'].mean()
    
    X = result_df.drop(columns=['CARBON_INTENSITY', 'CARBON_INTENSITY_LOG'])
    y = result_df['CARBON_INTENSITY_LOG']  # Train on log-transformed target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights.loc[X_train.index])

    # Predict and inverse transform
    y_pred_val_xgb = np.expm1(xgb_model.predict(validation_df.drop(columns=['CARBON_INTENSITY', 'CARBON_INTENSITY_LOG'])))

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=validation_df.index, y=validation_df['CARBON_INTENSITY'], label="True Values")
    sns.lineplot(x=validation_df.index, y=y_pred_val_xgb, label="XGBoost Predictions", alpha=0.2)
    y_pred_series = pd.Series(y_pred_val_xgb, index=validation_df.index)
    sns.lineplot(x=validation_df.index, y=y_pred_series.rolling(window=10, center=True).median(), label="XGBoost Predictions (Smoothed)")
    plt.xlabel("Timestamp")
    plt.ylabel("Carbon Intensity")
    plt.title("XGBoost: True vs Predicted Carbon Intensity")
    plt.savefig("xgb_carbon_intensity.png")

    # feature importance
    # Feature importance plot
    plt.figure(figsize=(10, 20))  # Adjust based on number of features
    importances = xgb_model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)
    feat_importances.plot(kind='barh')
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")


    

if __name__ == "__main__":
    main()
