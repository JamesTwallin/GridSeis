"""
UK Grid Frequency Analysis Tool
Analyzes and visualizes frequency data from the UK power grid.
"""

import datetime as dt
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import signal
from dateutil.relativedelta import relativedelta


class UKFrequencyAnalyzer:
    """Analyzes frequency data from the UK power grid."""
    
    def __init__(self, year: int, data_dir: Path = Path('data')):
        """
        Initialize the analyzer for a specific year.
        
        Args:
            year: The year to analyze
            data_dir: Directory containing the data files
        """
        self.year = year
        self.data_dir = data_dir
        self.frequency_data: Optional[pd.DataFrame] = None
        self.fft_df: Optional[pd.DataFrame] = None
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)

    def load_frequency_data(self) -> pd.DataFrame:
        """
        Load frequency data for the specified year.
        
        Returns:
            DataFrame containing the frequency data
        """
        print(f'Loading frequency data for year {self.year}')
        df = pd.read_csv(f'data/Frequency_{self.year}.csv')
        df['dtm'] = pd.to_datetime(df['dtm'])
        df.set_index('dtm', inplace=True)
        self.frequency_data = df
        return df

    def calculate_fft(self) -> pd.DataFrame:
        """
        Calculate Fast Fourier Transform for the frequency data.
        Processes data day by day to create a comprehensive FFT analysis.
        
        Returns:
            DataFrame containing the FFT results with periods as index
        """
        if self.frequency_data is None:
            self.load_frequency_data()

        # Determine date range based on year
        start_date = dt.datetime(year=self.year, month=1, day=1)
        end_date = (
            dt.datetime(self.year, month=7, day=30)
            if self.year == 2021
            else dt.datetime(self.year, month=12, day=31)
        )
        date_range = pd.date_range(start=start_date, end=end_date, freq='1D')

        # Initialize FFT DataFrame with periods as index
        fft_df = pd.DataFrame(index=np.arange(0, 3600, 0.1).round(1))
        
        # Calculate FFT for each day
        for day in date_range:
            print(f'Processing {day}')
            end = day + relativedelta(hours=24)
            mask = (self.frequency_data.index >= day) & (self.frequency_data.index < end)
            daily_data = self.frequency_data.loc[mask]
            
            # Calculate periodogram
            frequencies, power_spectrum = signal.periodogram(
                daily_data['f'], 
                fs=1.0, 
                scaling='spectrum'
            )
            
            # Convert frequencies to periods
            periods = 1 / frequencies
            power_spectrum = np.sqrt(power_spectrum)
            
            # Create temporary DataFrame and process
            temp_df = pd.DataFrame({
                'period': periods,
                'power': power_spectrum
            })
            temp_df = temp_df[temp_df['period'] < 3600].copy()
            temp_df['period'] = temp_df['period'].round(1)
            
            # Group by period and get maximum power
            grouped = temp_df.groupby('period')['power'].max()
            fft_df[day.strftime('%Y-%m-%d %H')] = grouped

        self.fft_df = fft_df
        output_file = self.data_dir / f'fft_{self.year}.csv'
        fft_df.to_csv(output_file)
        return fft_df

    def plot_frequency_data(self, n_samples: int = 1000000):
        """
        Plot raw frequency data.
        
        Args:
            n_samples: Number of samples to plot
        """
        if self.frequency_data is None:
            self.load_frequency_data()

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.frequency_data.head(n_samples))
        ax.tick_params(which='both', width=2)
        ax.set_ylim(49.5, 50.5)
        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel('Frequency (Hz)', fontsize=18)
        ax.set_title(f'{self.year} Frequency Data', fontsize=24)
        
        plt.tight_layout()
        plt.savefig(f'{self.year}_data.png')
        plt.close()

    def plot_dominant_oscillations(self):
        """Plot dominant oscillations from the frequency data."""
        if self.frequency_data is None:
            self.load_frequency_data()

        # Calculate periodogram
        frequencies, power_spectrum = signal.periodogram(
            self.frequency_data['f'],
            fs=1.0,
            scaling='spectrum'
        )
        
        # Process data
        periods = 1 / frequencies
        power_spectrum = np.power(power_spectrum, 0.1)
        
        df = pd.DataFrame({
            'period': periods,
            'power': power_spectrum
        })
        df = df[df['period'] < 3600].copy()
        df['period_rounded'] = df['period'].round(3)
        df_grouped = df.groupby('period_rounded')['power'].mean()

        # Create plot
        fig, ax = plt.subplots(figsize=(30, 10))
        ax.plot(df_grouped)
        ax.tick_params(which='both', width=2)
        ax.set_xlabel('Seconds', fontsize=14)
        ax.set_title(f'{self.year} dominant oscillations', fontsize=18)
        ax.set_xlim(0, 60)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(f'{self.year}_dominant_oscillations.png')
        plt.close()


def plot_frequency_histogram(data_dict: Dict[str, np.ndarray]):
    """
    Plot histogram of frequency data for multiple years.
    
    Args:
        data_dict: Dictionary mapping years to frequency data arrays
    """
    plt.figure(figsize=(9, 7))
    
    for year, frequencies in data_dict.items():
        sns.distplot(
            pd.Series(frequencies, name=year),
            label=year,
            kde=False,
            hist_kws={"histtype": "step", "linewidth": 2, "alpha": 1}
        )

    plt.legend()
    plt.xlim(49.6, 50.4)
    plt.yticks([])
    plt.xlabel('Frequency', fontsize=15)
    plt.tight_layout()
    plt.savefig('grid_histogram.png')
    plt.close()


def plot_fine_year_analysis(year: int, data_dir: Path):
    """
    Perform and plot detailed analysis for a specific year.
    
    Args:
        year: Year to analyze
        data_dir: Directory containing the data files
    """
    fft_file = data_dir / f'fft_{year}.csv'
    
    if not fft_file.exists():
        print(f"FFT data for {year} not found. Calculating FFT first...")
        analyzer = UKFrequencyAnalyzer(year)
        analyzer.calculate_fft()
        
    df = pd.read_csv(fft_file)
    df = df[df['Unnamed: 0'] < 30].copy()
    
    # Process timestamps
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], unit='s')
    df['Unnamed: 0'] = df['Unnamed: 0'].dt.strftime('%M:%S')
    df.set_index('Unnamed: 0', inplace=True)
    
    # Normalize data
    df = df.interpolate(method='linear')
    df = (df - df.mean()) / df.std()

    # Create plot
    plt.figure(figsize=(10, 10))
    sns.heatmap(df, cbar=False)
    plt.title('UK Grid Frequency Periodic Oscillations', fontsize=20)
    plt.xlabel('Date (YYYY-MM-DD HH)', fontsize=15)
    plt.ylabel('Period (minute:seconds)', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{year}.png')
    
    if year == 2015:
        plt.xlim(900, 1300)
        plt.ylim(590, 80)
        plt.savefig(f'zoomed_{year}.png')
    
    plt.close()


def join_frequency_data(base_path: str, years: List[int]) -> None:
    """
    Join frequency data from multiple files into yearly CSV files.
    
    Args:
        base_path: Base path to the frequency data files
        years: List of years to process
    """
    for year in years:
        year_df = pd.DataFrame()
        for month in range(1, 13):
            try:
                print(f'Processing year: {year} month: {month}')
                filepath = Path(base_path) / f'{year} {month}' / f'f {year} {month}.csv'
                
                # Print file existence and read first few lines
                if filepath.exists():
                    print(f"Found file: {filepath}")
                    with open(filepath, 'r') as f:
                        print("First few lines of file:")
                        for i, line in enumerate(f):
                            if i < 5:  # Print first 5 lines
                                print(line.strip())
                            else:
                                break
                else:
                    print(f"File not found: {filepath}")
                    continue
                
                df = pd.read_csv(filepath)
                print(f"Columns in file: {df.columns.tolist()}")
                
                # Handle different date formats with timezone information
                if year == 2014:
                    date_format = "%d/%m/%Y %H:%M:%S"
                else:
                    date_format = 'mixed'
                
                # Try to identify date column
                date_columns = [col for col in df.columns if col.lower() in ['dtm', 'date', 'datetime', 'timestamp']]
                if not date_columns:
                    print(f"No date column found in {filepath}")
                    continue
                    
                date_column = date_columns[0]
                print(f"Using {date_column} as date column")
                
                df.index = pd.to_datetime(df[date_column], format=date_format, utc=True).dt.tz_localize(None)
                df = df.drop(date_column, axis=1)
                year_df = pd.concat([year_df, df])
                
            except Exception as e:
                print(f'Error processing {year}-{month}: {e}')
        
        if not year_df.empty:
            output_path = Path('data') / f'Frequency_{year}.csv'
            output_path.parent.mkdir(exist_ok=True)
            year_df.to_csv(output_path)
            print(f"Saved data for year {year} to {output_path}")
        else:
            print(f"No data processed for year {year}")


def main():
    """Main execution function."""
    # Configuration
    PROCESS_ALL_DATA = False
    PLOT_HISTOGRAMS = True
    PLOT_FINE_YEARS = True
    BASE_PATH = Path('D:/Frequency data UK')
    DATA_DIR = Path('data')
    
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
    
    try:
        # First, try to join the raw frequency data
        print("Step 1: Processing raw frequency data...")
        # join_frequency_data(BASE_PATH, years=[2019, 2020, 2021])
        
        if PROCESS_ALL_DATA:
            print("\nStep 2: Calculating FFT for 2020...")
            analyzer = UKFrequencyAnalyzer(2020, data_dir=DATA_DIR)
            df = analyzer.calculate_fft()
            print("FFT calculation complete.")
        
        if PLOT_HISTOGRAMS:
            print("\nStep 3: Creating frequency histograms...")
            frequency_data = {}
            for year in range(2014, 2021):
                try:
                    yearly_analyzer = UKFrequencyAnalyzer(year, data_dir=DATA_DIR)
                    yearly_analyzer.load_frequency_data()
                    frequency_data[str(year)] = yearly_analyzer.frequency_data['f'].values
                    print(f"Processed data for {year}")
                except FileNotFoundError:
                    print(f"No data found for {year}, skipping...")
            
            if frequency_data:
                plot_frequency_histogram(frequency_data)
                print("Histogram plotting complete.")
        
        if PLOT_FINE_YEARS:
            print("\nStep 4: Creating detailed analysis for 2020...")
            try:
                analyzer = UKFrequencyAnalyzer(2020, data_dir=DATA_DIR)
                analyzer.load_frequency_data()
                analyzer.plot_frequency_data()
                analyzer.plot_dominant_oscillations()
                plot_fine_year_analysis(2020, DATA_DIR)
                print("Detailed analysis complete.")
            except Exception as e:
                print(f"Error in detailed analysis: {e}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nDebug information:")
        print(f"Working directory: {Path.cwd()}")
        print(f"Data directory: {DATA_DIR.absolute()}")
        print(f"Base path: {BASE_PATH}")
        print("\nAvailable files in data directory:")
        if DATA_DIR.exists():
            for file in DATA_DIR.glob('*'):
                print(f"  {file.name}")
        else:
            print("  Data directory does not exist")
    
    if PROCESS_ALL_DATA:
        # Process all years
        analyzer = UKFrequencyAnalyzer(2020, data_dir=DATA_DIR)
        df = analyzer.calculate_fft()
        
        # Join frequency data
        join_frequency_data(BASE_PATH, years=[2019, 2020, 2021])
    
    if PLOT_HISTOGRAMS:
        # Plot histograms for multiple years
        frequency_data = {}
        for year in range(2014, 2021):
            yearly_analyzer = UKFrequencyAnalyzer(year, data_dir=DATA_DIR)
            try:
                yearly_analyzer.load_frequency_data()
                frequency_data[str(year)] = yearly_analyzer.frequency_data['f'].values
            except FileNotFoundError as e:
                print(f"Skipping {year}: {e}")
    
    if PLOT_FINE_YEARS:
        # Detailed analysis for 2020
        analyzer = UKFrequencyAnalyzer(2020, data_dir=DATA_DIR)
        analyzer.load_frequency_data()
        analyzer.plot_frequency_data()
        analyzer.plot_dominant_oscillations()
        plot_fine_year_analysis(2020, DATA_DIR)


if __name__ == "__main__":
    main()