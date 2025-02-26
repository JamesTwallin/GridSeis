import pandas as pd
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

from helpers import get_raw_frequency_data, perform_fft_analysis, get_national_grid_data

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_combined_plot(fft_df_period, carbon_df_period, period_label, filename):
    """Create a combined plot of FFT heatmap and carbon intensity data for a given period."""
    fig = plt.figure(figsize=(15, 8))
    
    # Create a gridspec with better height ratios and spacing
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 0.1, 2], hspace=0.5)
    ax1 = fig.add_subplot(gs[0])  # FFT plot
    cbar_ax = fig.add_subplot(gs[1])  # Dedicated space for colorbar
    ax2 = fig.add_subplot(gs[2])  # Carbon intensity plot
    
    # Create heatmap
    hm = sns.heatmap(fft_df_period.T, cmap='viridis', ax=ax1, robust=True, cbar=False)
    
    # Flip the y-axis
    ax1.invert_yaxis()
    
    # Define a consistent left alignment position
    left_align = 0
    
    # Add labels and title
    ax1.set_xticklabels([])  # Remove x ticks from top plot
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title(f'Spectral Decomposition of Grid Frequency - 30 minute bins ({period_label})', 
                 ha='left', x=left_align, fontdict={'fontsize': 16})
    
    # Create a separate horizontal colorbar
    cbar_ax.set_position([0.3, cbar_ax.get_position().y0, 0.4, cbar_ax.get_position().height])
    
    norm = Normalize(vmin=hm.get_children()[0].get_array().min(), 
                    vmax=hm.get_children()[0].get_array().max())
    ColorbarBase(cbar_ax, cmap=plt.cm.viridis, norm=norm, orientation='horizontal', 
                label='Log Magnitude')
    
    # Plot carbon intensity 
    ax2.plot(range(len(carbon_df_period)), carbon_df_period['CARBON_INTENSITY'], 'k-', linewidth=1.5)
    
    # Fix the x-tick labels to match the dates (every 48th tick for daily)
    tick_indices = range(0, len(carbon_df_period), 48)
    ax2.set_title(f'National Grid Carbon Intensity ({period_label})', ha='left', x=left_align,
                fontdict={'fontsize': 16})
    ax2.set_xticks(tick_indices)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticklabels([pd.to_datetime(i).strftime('%Y-%m-%d') 
                        for i in carbon_df_period.index[tick_indices]], 
                        rotation=45, ha='right')
    
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Carbon Intensity (gCO2/kWh)')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.subplots_adjust(left=0.07, right=0.95, top=0.9, bottom=0.15)
    
    # Add suptitle aligned with subplot titles
    fig.suptitle('Grid Frequency Spectral Analysis and Carbon Intensity', 
                fontsize=20, fontweight='bold', x=0.07, y=0.98, ha='left')
    
    # Save the plot
    fig.savefig(filename, dpi=300)

if __name__ == "__main__":
    rows = []

    # Get carbon intensity data
    carbon_intensity_df = get_national_grid_data()
    carbon_intensity_df['DATETIME'] = pd.to_datetime(carbon_intensity_df['DATETIME'])
    carbon_intensity_df.set_index('DATETIME', inplace=True)
    carbon_intensity_df = carbon_intensity_df[['CARBON_INTENSITY']]
    
    # Process frequency data for each month
    for date in pd.date_range("2024-12-01", "2025-01-01", freq="1MS"):
        df = get_raw_frequency_data(date.year, date.month)
        if df is None or df.empty:
            continue
        
        # Process data in 30-minute blocks (1800 seconds)
        for i in range(0, len(df), 1800):
            try:
                timestamp = df.iloc[i]["dtm"]
                block = df.iloc[i:i+1800]
                fft_dict = perform_fft_analysis(block)
                fft_dict["timestamp"] = timestamp
                rows.append(fft_dict)
            except Exception as e:
                logging.error(f"Error processing block at index {i}: {e}")
    
    # Create DataFrame from FFT results
    fft_df = pd.DataFrame(rows)
    fft_df['timestamp'] = pd.to_datetime(fft_df['timestamp'])
    fft_df.set_index('timestamp', inplace=True)
    
    # Log transform the data
    fft_df = np.log1p(fft_df)
    
    # Format timestamps
    fft_df.index = fft_df.index.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Create full heatmap
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(111)
    
    # Generate the heatmap with y-axis inverted
    sns.heatmap(fft_df.T, cmap='viridis', ax=ax1, 
                robust=True, cbar_kws={'label': 'Log Magnitude'})
    
    # Flip the y-axis 
    ax1.invert_yaxis()
    
    # Add labels and title
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('FFT Analysis: Frequency Components Over Time')
    
    plt.tight_layout()
    fig.savefig('plots/fft_heatmap.png', dpi=300)
    
    # Process January data
    fft_df_jan = fft_df.loc['2025-01-22':'2025-01-31']
    carbon_df_jan = carbon_intensity_df.loc['2025-01-22':'2025-01-31']
    carbon_df_jan.index = carbon_df_jan.index.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Create January plot
    create_combined_plot(fft_df_jan, carbon_df_jan, 'Jan 22-31, 2025', 'plots/fft_heatmap_january.png')
    
    # Process December data
    fft_df_dec = fft_df.loc['2024-12-01':'2024-12-31']
    carbon_df_dec = carbon_intensity_df.loc['2024-12-01':'2024-12-31']
    carbon_df_dec.index = carbon_df_dec.index.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Create December plot
    create_combined_plot(fft_df_dec, carbon_df_dec, 'Dec 1-31, 2024', 'plots/fft_heatmap_december.png')