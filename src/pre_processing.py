import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time, bitfinex.bitfinex_v2

sns.set_style('darkgrid')
plt.style.use("fivethirtyeight")

TIMEFRAME = '4h'
TIMEFRAME_IN_S = 4 * 3600
TICKER_LIST = ['BTCUSD', 'BTCJPY', 'BTCEUR', 'ETHUSD', 'XRPUSD', 'doge:usd']

class BTCUSD_Features():
    # For multiple API calls to Bitfinex API
    @staticmethod
    def fetch_data(start, stop, ticker, interval, TIMEFRAME_S):
        # Maximum 1000 data points per call
        limit = 1000  

        # Create Bitfinex api instance
        api_v2 = bitfinex.bitfinex_v2.api_v2()
        hour = TIMEFRAME_S * 1000
        step = hour * limit
        data = []

        total_steps = (stop-start)/hour
        while total_steps > 0:
            if total_steps < limit:
                step = total_steps * hour

            end = start + step
            data += api_v2.candles(symbol=ticker, interval=interval, limit=limit, start=start, end=end)
            print(pd.to_datetime(start, unit='ms'), pd.to_datetime(end, unit='ms'), "steps left:", total_steps)
            start = start + step
            total_steps -= limit
            time.sleep(1.5)
        return data
    
    def create_features(self):
        # Creates timestamps and converts them to Unix Timestamp (seconds from Unix Epoch)
        time_end_datetime = datetime.now()
        time_start_datetime = datetime(time_end_datetime.year - 1, time_end_datetime.month, time_end_datetime.day)
        time_end = time.mktime(time_end_datetime.timetuple()) * 1000
        time_start = time.mktime(time_start_datetime.timetuple()) * 1000

        ticker_data = {}

        for ticker in TICKER_LIST:   
            result = self.fetch_data(time_start, time_end, ticker, TIMEFRAME, TIMEFRAME_IN_S)
            names = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
            
            ticker_data[ticker] = pd.DataFrame(result, columns=names)
            ticker_data[ticker]['Date'] = pd.to_datetime(ticker_data[ticker]['Date'], unit='ms')
            ticker_data[ticker].set_index('Date', inplace=True)
            ticker_data[ticker] = ticker_data[ticker].sort_values(by='Date')

        return ticker_data
    
    def merge_features(self, features_dictionary):
        # Resample the ETHUSD, XRPUSD, and doge:usd dataframes to a 4-hour frequency
        eth_high = features_dictionary['ETHUSD'].resample('4H')['High'].max()
        xrp_high = features_dictionary['XRPUSD'].resample('4H')['High'].max()
        doge_high = features_dictionary['doge:usd'].resample('4H')['High'].max()
        btcjpy_high = features_dictionary['BTCJPY'].resample('4H')['High'].max()
        btceur_high = features_dictionary['BTCEUR'].resample('4H')['High'].max()
        btcusd_all = features_dictionary['BTCEUR'].resample('4H').max()

        # Combine the resampled dataframes into a single dataframe
        high_df = pd.concat([btcjpy_high, btceur_high, eth_high, xrp_high, doge_high], axis=1)
        high_df.columns = ['BTCPJPY', 'BTCEUR' ,'ETHUSD', 'XRPUSD', 'DOGEUSD']
        btcusd_cleaned = btcusd_all.drop_duplicates()

        combined_df = pd.concat([btcusd_cleaned, high_df], axis=1)

        return combined_df
    
    def get_moving_avgs_df(self, dataframe, length1, length2, length3):
        ma_lengths = [length1, length2, length3]
        features = ['High', 'Low', 'BTCPJPY', 'BTCEUR', 'ETHUSD', 'XRPUSD', 'DOGEUSD']
        moving_avgs = pd.DataFrame(columns=[f"{ma} Candle MA for {column}" for ma in ma_lengths for column in features])
        # print(BTCUSD_df.to_string())
        for ma in ma_lengths:
            for column in features:
                column_name = f"{ma} Candle MA for {column}"
                moving_avgs[column_name] = dataframe[column].rolling(ma).mean()
    
    def get_full_df(self):
        ticker_data = self.create_features()
        processed_df = self.merge_features(ticker_data)
        return processed_df