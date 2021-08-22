import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

class UKFrequency():
    def __init__(self, year):
        self.frequency_data = None
        self.fft_df = None
        self.monthly_fft = None
        self.twenty_fifteen_fft = None
        self.year = year

    def load_frequency_data(self):
        print('loading frequency data for year {}'.format(self.year))
        df = pd.read_csv('Frequency_{}.csv'.format(self.year))
        print(df.head())
        df['dtm'] = pd.to_datetime(df['dtm'])
        print('setting index')
        df.set_index('dtm', inplace=True)
        self.frequency_data = df
        return df


    def fft(self):
        if self.frequency_data is None:
            self.load_frequency_data()
        if self.year == 2021:
            date_range = pd.date_range(start=dt.datetime(year=self.year, month=1, day=1),
                                       end=dt.datetime(self.year, month=7, day=30), freq='1D')
        else:
            date_range = pd.date_range(start=dt.datetime(year=self.year, month=1, day=1),
                                       end=dt.datetime(self.year, month=12, day=31), freq='1D')
        fft_df = pd.DataFrame(index=np.arange(0, 3600., .1).round(1))
        for day in date_range:
            print(day)
            end = day + relativedelta(hours=24)
            filt = ((self.frequency_data.index >= day) & (self.frequency_data.index < end))
            df = self.frequency_data.loc[filt]
            y = df['f']
            fs = 1.
            f, Pxx = signal.periodogram(y, fs=fs, scaling='spectrum')
            f = (1 / f)
            Pxx = np.sqrt(Pxx)
            df = pd.DataFrame()
            df['f'] = f
            df['p'] = Pxx
            filt = df['f'] < 3600.
            df = df.loc[filt]
            df = df.reset_index()
            df = df.drop('index', axis=1)
            df['f'] = df['f'].round(1)
            a = df.groupby('f').max()
            fft_df['{}'.format(day.strftime('%Y-%m-%d %H'))] = a

        self.fft_df = fft_df
        fft_df.to_csv('fft_{}.csv'.format(self.year))
        return fft_df






def join_frequency_data():
    years = [2019, 2020,2021]
    months = np.arange(1, 13, 1)
    for year in years:
        yeardf = pd.DataFrame()
        for month in months:
            try:
                print('year:{y} month{m}'.format(y=year, m=month))
                df = pd.read_csv(
                    r'D:\Frequency data UK\{y} {m}\f {y} {m}.csv'.format(y=year, m=month))
                if year == 2014:
                    pass
                    df.index = pd.to_datetime(df['dtm'], format="%d/%m/%Y %H:%M:%S", utc=True).dt.tz_localize(None)
                else:
                    df.index = pd.to_datetime(df['dtm'], format="%Y-%m-%d %H:%M:%S", utc=True).dt.tz_localize(None)

                df = df.drop('dtm', axis=1)
                yeardf = yeardf.append(df)
            except Exception as e:
                print(e)

        yeardf.to_csv('Frequency_{}.csv'.format(year))





def plot_histogram(dict):
    plt.figure(figsize=(9, 7))
    for key, value in dict.items():
        x = pd.Series(value, name=key)
        ax = sns.distplot(x, label=key, kde=False, hist_kws={"histtype": "step", "linewidth": 2, "alpha": 1, })

    plt.legend()
    plt.xlim(49.6, 50.4)
    plt.yticks([])
    plt.xlabel('Frequency', fontsize=15)
    plt.savefig('grid_histogram')
    plt.tight_layout()
    plt.close()


if __name__ == "__main__":
    # join_frequency_data()

    plot_all_data = False
    plot_histograms = False
    plot_1d = True
    off_line = False
    off_line_daily = False
    plot_fine_years = True
    eclipse = False


    if plot_all_data:
        all_data = pd.DataFrame()
        count = 1
        for year in [2020]:
            test = UKFrequency(year=year)
            # df = test.do_monthly_fft()
            df = test.fft()
            # test.plot_sub_minute()
            if count == 1:
                all_data = all_data.append(df)
            else:
                all_data = pd.merge(all_data, df, right_index=True, left_index=True)
            count += 1
        all_data.to_csv('all_daily_hourly_data.csv')

    if plot_histograms:
        dict = {}
        for year in [2014, 2015, 2016, 2017, 2018, 2019,2020]:
            test = UKFrequency(year)
            test.load_frequency_data()
            dict['{}'.format(year)] = test.frequency_data['f'].values
        plot_histogram(dict)

    if off_line:
        df = pd.read_csv('all_daily_hourly_data.csv')
        filt = ((df['Unnamed: 0'] >= 0.)&(df['Unnamed: 0'] != 60.) &(df['Unnamed: 0'] <= 90.))
        df = df.loc[filt]
        df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], unit='s')
        df['Unnamed: 0'] = df['Unnamed: 0'].dt.strftime('%M:%S')
        df.index = df['Unnamed: 0']
        df = df.fillna(method='ffill')
        df = df.drop('Unnamed: 0', axis=1)
        # normalized_df = (df - df.mean()) / df.std()

        plt.show()

        fig = plt.figure(figsize=(20, 7))
        ax = sns.heatmap(df, cbar=False)

        plt.title('UK Grid Frequency Periodic Oscillations by Month', fontsize=20)  # title with fontsize 20
        plt.xlabel('Months', fontsize=15)  # x-axis label with fontsize 15
        plt.ylabel('Period (minute:seconds)', fontsize=15)

        plt.tight_layout()
        plt.savefig('off_line_10minute.png')
        plt.close()

    if off_line_daily:
        df = pd.read_csv('all_daily_hourly_data.csv')
        # filt = ((df['Unnamed: 0'] < 90.) & (df['Unnamed: 0'] != 60.)&(df['Unnamed: 0'] != 150.))
        filt = (df['Unnamed: 0'] < 60.) & (df['Unnamed: 0'] != 60.)& (df['Unnamed: 0'] != 29.)& (df['Unnamed: 0'] != 30.)
        df = df.loc[filt]
        df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], unit='s')
        df['Unnamed: 0'] = df['Unnamed: 0'].dt.strftime('%M:%S')
        df.index = df['Unnamed: 0']
        df = df.drop('Unnamed: 0', axis=1)
        df = df.interpolate(method='linear')
        # df = df.T.rolling(5,center=True).mean()
        # df = df.T
        plt.show()
        fig = plt.figure(figsize=(20, 7))
        ax = sns.heatmap(df, cbar=False)
        plt.title('UK Grid Frequency Periodic Oscillations', fontsize=20)  # title with fontsize 20
        plt.xlabel('Date', fontsize=15)  # x-axis label with fontsize 15
        plt.ylabel('Period (minute:seconds)', fontsize=15)

        plt.tight_layout()
        plt.savefig('off_line_hourly_10minute.png')
        plt.close()

    if plot_fine_years:
        for year in [2020]:
            print(year)
            df = pd.read_csv('fft_{}.csv'.format(year))
            filt = (df['Unnamed: 0'] < 30.)
            df = df.loc[filt]
            df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], unit='s')
            df['Unnamed: 0'] = df['Unnamed: 0'].dt.strftime('%M:%S')
            df.index = df['Unnamed: 0']
            df = df.drop('Unnamed: 0', axis=1)
            df = df.interpolate(method='linear')
            # df = df.T.rolling(5).sum()
            # df = df.T.rolling(3).sum()
            df = (df - df.mean()) / df.std()
            plt.show()

            fig = plt.figure(figsize=(10, 10))
            ax = sns.heatmap(df, cbar=False)
            if year == 2015:
                plt.title('UK Grid Frequency Periodic Oscillations', fontsize=20)  # title with fontsize 20
                plt.xlabel('Date (YYYY-MM-DD HH)', fontsize=15)  # x-axis label with fontsize 15
                plt.ylabel('Period (minute:seconds)', fontsize=15)

                plt.tight_layout()
                plt.savefig('{}.png'.format(year))

                ax.set_xlim(900, 1300)
                ax.set_ylim(590., 80.)

                plt.savefig('zoomed_{}.png'.format(year))
                plt.close()
            else:
                plt.title('UK Grid Frequency Periodic Oscillations', fontsize=20)  # title with fontsize 20
                plt.xlabel('Date (YYYY-MM-DD HH)', fontsize=15)  # x-axis label with fontsize 15
                plt.ylabel('Period (minute:seconds)', fontsize=15)

                plt.tight_layout()
                plt.savefig('{}.png'.format(year))
                plt.close()

    if plot_1d:
        test = UKFrequency(2020)
        df = test.load_frequency_data()

        # fig, ax = plt.subplots(figsize=(20, 10))
        #
        # ax.plot(df.head(1000000))
        # ax.tick_params(which='both', width=2)
        # ax.set_ylim(49.5, 50.5)
        # ax.set_xlabel('Time', fontsize=18)
        # ax.set_ylabel('Frequency (Hz)', fontsize=18)
        # ax.set_title('2020 Frequency Data', fontsize=24)
        #
        # fig.tight_layout()
        # fig.savefig('2020_data.png')
        # plt.close()

        y = df['f']
        fs = 1.
        f, Pxx = signal.periodogram(y, fs=fs, scaling='spectrum')
        f = np.divide(1. ,f)
        # push up the lower values
        Pxx = np.power(Pxx,0.1)
        # f = np.divide(f,60.)
        df = pd.DataFrame()
        df['f'] = f
        df['p'] = Pxx
        filt = df['f'] < 3600.
        df = df.loc[filt]
        df['f_rounded'] = df['f'].round(3)
        df2 = df[['f_rounded', 'p']].groupby('f_rounded').mean()
        fig, ax = plt.subplots(figsize=(30, 10))
        ax.plot(df2['p'])
        ax.tick_params(which='both', width=2)
        ax.set_xlabel('Seconds', fontsize=14)
        ax.set_title('2020 dominant oscillations', fontsize=18)
        ax.set_xlim(0, 60.)
        # minor_ticks = np.arange(5, 61, 10)
        # ax.set_xticks(minor_ticks, minor=True)
        # ax.set_xticklabels(minor=True, labels=minor_ticks)

        # And a corresponding grid

        # Or if you want different settings for the grids:
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.set_yticks([])

        plt.show()

        fig.tight_layout()
        fig.savefig('2020_1d_plot.png')
        plt.close()
