import pandas as pd
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.model_selection import train_test_split
import xgboost as xgb

from helpers import get_raw_frequency_data, perform_fft_analysis, get_national_grid_data

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    rows = []

    carbon_intensity_df = get_national_grid_data()
    carbon_intensity_df['DATETIME'] = pd.to_datetime(carbon_intensity_df['DATETIME'])
    carbon_intensity_df.set_index('DATETIME', inplace=True)
    carbon_intensity_df = carbon_intensity_df[['CARBON_INTENSITY']]
    
    for date in pd.date_range("2024-12-01", "2025-01-01", freq="1MS"):
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

    # make 2d plot
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(111)
    
    # Log scale it
    fft_df = np.log1p(fft_df)
    
    # Convert the timestamps to %Y-%m-%dT%H:%M:%SZ
    fft_df.index = fft_df.index.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Generate the heatmap with y-axis inverted
    sns.heatmap(fft_df.T, cmap='viridis', ax=ax1, 
                robust=True, cbar_kws={'label': 'Log Magnitude'})
    
    # Flip the y-axis to have lower period values at the bottom
    ax1.invert_yaxis()
    
    # Add labels and title
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('FFT Analysis: Frequency Components Over Time')
    
    plt.tight_layout()

    # Save the plot
    fig.savefig('fft_heatmap.png', dpi=300)


    fig = plt.figure(figsize=(30, 16))  # Made taller to accommodate colorbar

    # Create a gridspec with specified height ratios and proper spacing
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 0.2, 1], hspace=0.2)
    ax1 = fig.add_subplot(gs[0])  # FFT plot
    cbar_ax = fig.add_subplot(gs[1])  # Dedicated space for colorbar
    ax2 = fig.add_subplot(gs[2])  # Carbon intensity plot

    # Get the last 10 days of January
    fft_df_jan = fft_df.loc['2025-01-22':'2025-01-31']

    # Get the carbon intensity data and format it to match
    carbon_df_jan = carbon_intensity_df.loc['2025-01-22':'2025-01-31']
    carbon_df_jan.index = carbon_df_jan.index.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Create heatmap for January data
    hm = sns.heatmap(fft_df_jan.T, cmap='viridis', ax=ax1, 
                robust=True, cbar=False)  # No colorbar here

    # Flip the y-axis for January plot
    ax1.invert_yaxis()

    # Add labels and title for January plot
    ax1.set_xticklabels([])  # Remove x ticks from top plot
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('FFT Analysis: Frequency Components (Jan 22-31, 2025)')

    # Create a separate horizontal colorbar in the middle subplot
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=hm.get_children()[0].get_array().min(), 
                    vmax=hm.get_children()[0].get_array().max())
    ColorbarBase(cbar_ax, cmap=plt.cm.viridis, norm=norm, orientation='horizontal', 
                label='Log Magnitude')

    # Plot carbon intensity 
    ax2.plot(range(len(carbon_df_jan)), carbon_df_jan['CARBON_INTENSITY'], 'k-', linewidth=1.5)

    # Fix the x-tick labels to match the dates
    ax2.set_xticks(range(len(carbon_df_jan)))
    ax2.set_xticklabels(carbon_df_jan.index, rotation=45, ha='right')

    # every 48th
    tick_indices = range(0, len(carbon_df_jan), 48)
    ax2.set_title('Carbon Intensity (Jan 22-31, 2025)')
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels([carbon_df_jan.index[i] for i in tick_indices], rotation=45, ha='right')

    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Carbon Intensity')
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    
    # Adjust subplot positions manually to ensure alignment
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1)

    # Save the January plot
    fig.savefig('fft_heatmap_january.png', dpi=300)