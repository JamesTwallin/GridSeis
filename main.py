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
        # if freq > 0.001 and freq < 0.5:  # 0-100mHz
            # if 1/freq == 10 or 1/freq == 5 or 1/freq == 7.5:
            #     continue
            if 1/freq > 6 or 1/freq < 3: 
                continue
            period_dict[f"period_{1/freq:.4f}"] = mag
   
   return period_dict

def main():
    rows = []
    for date in pd.date_range("2024-10-01", "2025-01-01", freq="1MS"):
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
        
    fft_columns = [col for col in result_df.columns if col.startswith('period_')]

    # # Create new features in separate DataFrames
    # shifted_df = pd.DataFrame({
    #     f'{col}_shift_5': result_df[col].shift(5)
    #     for col in fft_columns
    # })

    rolling_mean_df = pd.DataFrame({
        f'{col}_{window}_mean': result_df[col].rolling(window).mean()
        for col in fft_columns
        for window in ['1h', '3h', '6h']
    })

    rolling_max_df = pd.DataFrame({
        f'{col}_{window}_max': result_df[col].rolling(window).max()
        for col in fft_columns
        for window in ['1h', '3h', '6h']
    })



    # Drop original columns and combine with new features
    result_df = result_df.drop(columns=fft_columns)
    result_df = pd.concat([result_df, rolling_mean_df, rolling_max_df], axis=1)






    
    fuel_data = get_national_grid_data()
    if fuel_data is None:
        logging.error("Failed to load fuel data. Exiting.")
        return
    
    fuel_data['DATETIME'] = pd.to_datetime(fuel_data['DATETIME'])
    fuel_data.set_index('DATETIME', inplace=True)
    fuel_data = fuel_data[['CARBON_INTENSITY']]
    
    result_df['CARBON_INTENSITY'] = fuel_data['CARBON_INTENSITY']
    result_df.dropna(inplace=True)
    result_df.drop_duplicates(inplace=True)



    validation_df = result_df[result_df.index.year == 2025]
    result_df = result_df[result_df.index.year != 2025]

    X = result_df.drop(columns=['CARBON_INTENSITY'])
    y = result_df['CARBON_INTENSITY']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_model = xgb.XGBRegressor(
        # use gpu
        tree_method='gpu_hist',
        gpu_id=0,
    )

    xgb_model.fit(X_train, y_train) 

    y_pred_val_xgb = xgb_model.predict(validation_df.drop(columns=['CARBON_INTENSITY']))
    # make a series

    plt.figure(figsize=(20, 6))
    sns.lineplot(x=validation_df.index, y=validation_df['CARBON_INTENSITY'], label="True Values")
    sns.lineplot(x=validation_df.index, y=y_pred_val_xgb, label="XGBoost Predictions", alpha=0.2)
    # plot the max and min of the predictions

    # smooth the plot with a Savitzky-Golay filter
    y_pred_val_xgb_smooth = signal.savgol_filter(y_pred_val_xgb, 51, 3)
    sns.lineplot(x=validation_df.index, y=y_pred_val_xgb_smooth, label="XGBoost Predictions (Smoothed)")


    plt.xlabel("Timestamp")
    plt.ylabel("Carbon Intensity")
    plt.title("XGBoost: True vs Predicted Carbon Intensity")
    plt.savefig("xgb_carbon_intensity.png")

    # feature importance
    # Feature importance plot
    plt.figure(figsize=(10, 200))  # Adjust based on number of features
    importances = xgb_model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns)
    feat_importances.plot(kind='barh')
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")

    # scatter
    y_pred_test_xgb = xgb_model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create scatter plot
    ax.scatter(y_test, y_pred_test_xgb, color="#16BAC5", alpha=0.5, zorder=2)
    # plot a 1:1
    ax.plot([0, 300], [0, 300], color='gray', linestyle='--', zorder=3)
    

    # Configure axes
    ax.set_xlabel("Actual GB Carbon Intensity (gCO2/kWh)")
    ax.set_ylabel("Predicted GB Carbon Intensity (gCO2/kWh)")

    # Remove top and right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Add grid
    ax.yaxis.grid(color='gray', linestyle='dashed', zorder=0)
    ax.xaxis.grid(color='gray', linestyle='dashed', zorder=0)

    # Set title above the figure
    fig.suptitle("Grid frequency data can be used to predict carbon intensity",
                x=0.02,
                horizontalalignment='left',
                verticalalignment='bottom',
                fontsize=12,
                fontweight='bold',
                transform=fig.transFigure)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("scatter_xgb.png", dpi=300, bbox_inches='tight')





    

if __name__ == "__main__":
    main()
