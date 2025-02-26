import pandas as pd
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

# Machine Learning
from sklearn.model_selection import train_test_split
import xgboost as xgb

from helpers import get_raw_frequency_data, perform_fft_analysis, get_national_grid_data

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def prepare_fft_data(start_date, end_date):
    """Process raw frequency data and extract FFT features"""
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    rows = []
    for date in pd.date_range(start_date, end_date, freq="1MS"):
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
    
    fft_df = pd.DataFrame(rows)
    fft_df['timestamp'] = pd.to_datetime(fft_df['timestamp'])
    fft_df.set_index('timestamp', inplace=True)
    
    return fft_df


def add_features(fft_df):
    """Add derived features to the dataset"""
    fft_columns = fft_df.columns
    
    # Create rolling window features
    rolling_max_df = pd.DataFrame({
        f'{col}_{window}_max': fft_df[col].rolling(window).max()
        for col in fft_columns
        for window in ['1h', '3h', '6h']
    })

    rolling_min_df = pd.DataFrame({
        f'{col}_{window}_min': fft_df[col].rolling(window).min()
        for col in fft_columns
        for window in ['1h', '3h', '6h']
    })

    # Add cyclical time features
    fft_df['sin_second'] = np.sin(2 * np.pi * fft_df.index.second / 86400)
    fft_df['cos_second'] = np.cos(2 * np.pi * fft_df.index.second / 86400)

    # Combine all features
    fft_df = pd.concat([fft_df, rolling_min_df, rolling_max_df], axis=1)
    
    return fft_df


def merge_with_carbon_data(fft_df):
    """Merge FFT data with carbon intensity data"""
    fuel_data = get_national_grid_data()
    if fuel_data is None:
        logging.error("Failed to load fuel data.")
        return None
    
    fuel_data['DATETIME'] = pd.to_datetime(fuel_data['DATETIME'])
    fuel_data.set_index('DATETIME', inplace=True)
    fuel_data = fuel_data[['CARBON_INTENSITY']]
    
    fft_df['CARBON_INTENSITY'] = fuel_data['CARBON_INTENSITY']
    fft_df.dropna(inplace=True)
    fft_df.drop_duplicates(inplace=True)
    
    return fft_df


def train_model(fft_df, validation_date):
    if not os.path.exists("models"):
        os.makedirs("models")
    # get the pickle file

    """Train XGBoost model and split data"""
    validation_df = fft_df[fft_df.index > validation_date]
    training_df = fft_df[fft_df.index <= validation_date]

    X = training_df.drop(columns=['CARBON_INTENSITY'])
    y = training_df['CARBON_INTENSITY']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if os.path.exists("models/xgb_model.pkl"):
        with open("models/xgb_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)
    else:
        xgb_model = xgb.XGBRegressor(device="cuda")
        xgb_model.fit(X_train, y_train)
        with open("models/xgb_model.pkl", "wb") as f:
            pickle.dump(xgb_model, f)

    return xgb_model, X_test, y_test, validation_df


def plot_validation_results(validation_df, y_pred_val):
    """Create validation plots with original and detrended data"""
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # Original data plot
    ax1.plot(validation_df.index, validation_df['CARBON_INTENSITY'], label="True Values", color='black')
    ax1.plot(validation_df.index, y_pred_val, label="Predictions using frequency data", alpha=0.5, color='red')
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Carbon Intensity (gCO2/kWh)")
    
    # Detrended data plot
    ax2.plot(validation_df.index, validation_df['CARBON_INTENSITY'] - validation_df['CARBON_INTENSITY'].rolling('12h').mean(), 
             label="True Values", color='black')
    
    # Convert to series for rolling calculations
    y_pred_val_series = pd.Series(y_pred_val, index=validation_df.index)
    corrected = y_pred_val_series - y_pred_val_series.rolling('12h').mean()
    smoothed = corrected.rolling('1h').mean()
    ax2.plot(validation_df.index, smoothed, label="Predictions using frequency data", alpha=0.5, color='red')

    ax2.set_xlabel("Timestamp")
    ax2.set_ylabel("Carbon Intensity (Detrended)")

    text = f'Predictions for {validation_df.index[0].strftime("%B %Y")}'

    ax1.set_title(text,  
                loc='left',
                x=0.02,
                y=0.9,
                transform=fig.transFigure)
    
    ax2.set_title("Detrended Carbon Intensity - local minima are captured",
                loc='left',
                x=0.02,
                y=0.40,
                transform=fig.transFigure)
    
    fig.suptitle("FREQUENCY DATA CONTAINS INFORMATION ABOUT CARBON INTENSITY",
                x=0.02,
                y=0.93, 
                horizontalalignment='left', 
                verticalalignment='bottom',
                fontsize=16,
                fontweight='bold',
                transform=fig.transFigure)

    # Clean up plot aesthetics
    for ax in [ax1, ax2]:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    # rotate x-axis labels
    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

    # add a legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    plt.subplots_adjust(left=0.07, right=0.95, top=0.87, bottom=0.1, hspace=0.6)
    plt.savefig("plots/validation.png")

def plot_scatter_comparison(y_test, y_pred_test):
    """Create scatter plot comparing actual vs predicted values"""
    fig, ax = plt.subplots(figsize=(6, 6))

    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred_test)
    r2 = r_value**2

    # Create scatter plot
    ax.scatter(y_test, y_pred_test, color="red", alpha=0.5, zorder=2, s=1, marker='.')
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
     # Add R-squared text
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            fontsize=12, fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))

    # Set title
    fig.suptitle("FREQUENCY DATA CONTAINS INFORMATION ABOUT CARBON INTENSITY",
                x=0.02, horizontalalignment='left', verticalalignment='bottom',
                fontsize=10, fontweight='bold', transform=fig.transFigure)
    
    ax.set_title("Training a model using only frequency data provides good predictions",
            loc='left',
            x=0.02,
            y=0.98,
            fontdict={'fontsize': 8},
            transform=fig.transFigure)

    plt.tight_layout()
    plt.savefig("plots/scatter_xgb.png", dpi=300, bbox_inches='tight')


def main():
    # Process steps
    fft_df = prepare_fft_data("2023-07-01", "2024-12-01")
    fft_df = add_features(fft_df)
    fft_df = merge_with_carbon_data(fft_df)
    
    if fft_df is None:
        logging.error("Failed to prepare dataset. Exiting.")
        return
    
    # Train model and generate predictions
    model, X_test, y_test, validation_df = train_model(fft_df, "2024-12-01")
    y_pred_val = model.predict(validation_df.drop(columns=['CARBON_INTENSITY']))
    y_pred_test = model.predict(X_test)
    
    # Create plots
    plot_validation_results(validation_df, y_pred_val)
    plot_scatter_comparison(y_test, y_pred_test)


if __name__ == "__main__":
    main()