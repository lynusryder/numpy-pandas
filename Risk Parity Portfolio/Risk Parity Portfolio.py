import os
import yfinance as yf
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")


class Strategy_Risk_Parity:
    
    def __init__(self):
        
        self.directory = "C:\\Users\\Linus Ong\\Downloads"
        
        self.start_date = '2004-11-18'
        self.end_date = '2023-03-31'
        self.freq = '1d'
        self.to_resample = '1m'
        
        sr_dict = {'1d': np.sqrt(252),
                        '1m': np.sqrt(12),
                        '1y': np.sqrt(1)}
        self.sr = sr_dict[self.to_resample]
        
        self.test_size = 0.4
        self.hist_best_period = 6
        self.best_exp_period = 2
        
        try:
            data = pd.read_csv("C:\\Users\\Linus Ong\\Desktop\\data.csv") 
            data.set_index('Date', inplace=True)
            data.index = pd.to_datetime(data.index)
            self.data = data.pct_change().dropna()
            if self.to_resample == '1m':
                self.data = self.data.resample('M').sum()
            print("Data retrieved.")
        except:
            self.get_data()

        self.train_test_split()
        
    def save_data(self, df):
        
        df.to_csv("C:\\Users\\Linus Ong\\Desktop\\data.csv")

    def get_data(self):
        
        stocks = ['TLT', 'GLD', 'SPY']
        data = pd.DataFrame()
        for i in tqdm(range(0, len(stocks))):
            stock = stocks[i]
            ticker = yf.Ticker(stock)
            historical_data = ticker.history(start=self.start_date, end=self.end_date, interval=self.freq)
            historical_data = historical_data[["Close"]]
            historical_data.rename(columns = {'Close': stock}, inplace = True)
            if data.empty:
                data = historical_data.copy()
            else:
                data = data.join(historical_data)
        
        data = data.ffill().bfill()
#        print(data.head())
#        data.set_index('Date', inplace=True)
        self.data = data.pct_change().dropna()
        if self.to_resample == '1m':
            self.data = self.data.resample('M').sum()
        try:
            self.save_data(data)
            print("Data saved.")
        except:
            print("Error saving data.")
            
    def train_test_split(self):
        
        self.train_data, self.test_data = self.data[:round((1-self.test_size)*len(self.data))], self.data[round((1-self.test_size)*len(self.data)):]
        
    def sharpe_ratio(self, y):
        
        return np.sqrt(126) * (y.mean() / y.std())
        
    def historical_vol(self): 
        
        days_result_dict = pd.DataFrame(columns = ['window', 'total_return'])
        for window in tqdm(range(2, 12)):
            
            df = self.train_data
            df = df.rolling(window).std().shift(1)
            df = 1/df
            df['sum_of_stds'] = df.sum(axis=1)
            weights_df = df.loc[:,"TLT":"SPY"].div(df["sum_of_stds"], axis=0)
            excluded_days = df.index.values[:window-1] 
            eda = df.index.isin(excluded_days)
            port_ret = weights_df.mul(self.train_data[~eda])
            port_ret['daily_returns'] = port_ret.sum(axis=1)
            port_ret['cumulative_returns'] = port_ret['daily_returns'].cumsum()
            
            d_ = {'window': window, 'total_return': port_ret['cumulative_returns'].values[-1]}
            days_result_dict = days_result_dict.append(d_, ignore_index=True)
        
        max_return = np.max(days_result_dict['total_return'])
        self.results_table = days_result_dict
        self.hist_best_period = int(days_result_dict[days_result_dict['total_return']==max_return].window.values[0])
        
        best_window = self.hist_best_period
        df = self.train_data
        df = df.rolling(best_window).std().shift(1)
        df = 1/df
        df['sum_of_stds'] = df.sum(axis=1)
        weights_df = df.loc[:,"TLT":"SPY"].div(df["sum_of_stds"], axis=0)
        excluded_days = df.index.values[:window-1] 
        eda = df.index.isin(excluded_days)
        port_ret = weights_df.mul(self.train_data[~eda])
        port_ret['daily_returns'] = port_ret.sum(axis=1)
        port_ret['histV_cumulative_returns'] = port_ret['daily_returns'].cumsum()
        
        train_data_for_eqw = self.train_data[~eda].copy()
        train_data_for_eqw['TLT'] = train_data_for_eqw['TLT'] * 0.35
        train_data_for_eqw['SPY'] = train_data_for_eqw['SPY'] * 0.55
        train_data_for_eqw['GLD'] = train_data_for_eqw['GLD'] * 0.1
        train_data_for_eqw['daily_returns'] = train_data_for_eqw.sum(axis=1)
        train_data_for_eqw['stdW_cumulative_returns'] = train_data_for_eqw['daily_returns'].cumsum()
        
        df_combined = train_data_for_eqw[['stdW_cumulative_returns']].join(port_ret[['histV_cumulative_returns']])
        df_combined.plot()
        plt.title("Historical Vol Weighted vs Standard Weights: Training Data Set")
        plt.xticks(rotation=70)
        plt.show()
        
        sharpe_ratio_of_stdW = self.sr*np.mean(train_data_for_eqw['daily_returns'])/np.std(train_data_for_eqw['daily_returns'])
        sharpe_ratio_of_histV = self.sr*np.mean(port_ret['daily_returns'])/np.std(port_ret['daily_returns'])
        print("Optimized rolling window period is:", self.hist_best_period)
        print("Sharpe Ratio of Standard Weighting:", sharpe_ratio_of_stdW)
        print("Sharpe Ratio of Historical Risk Parity Weighting:", sharpe_ratio_of_histV)
        
        print("Cumulative returns of HistoricalVol Strategy:", df_combined.histV_cumulative_returns.values[-1])
        print("Cumulative returns of Standard Weighting Strategy:", df_combined.stdW_cumulative_returns.values[-1])
        
    def garch_model(self, x):
        
        m = arch_model(x, p = 1, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 'normal')
        
        result = m.fit(disp='off')
        forecast = result.forecast(horizon=1)
        
        return np.sqrt(forecast.variance.values[-1][0])
        
    def expected_vol(self):
        
        days_result_dict = pd.DataFrame(columns = ['window', 'total_return'])
        for window in tqdm(range(2, 12)):
            
            expected_vol_df = pd.DataFrame(columns=self.train_data.columns, index=self.train_data.index)
            for row in range(0, len(expected_vol_df)):
                
                sliced_df = self.train_data.iloc[row:row+window]*100
                d_ = {}
                for column in sliced_df.columns:
                    forecast = self.garch_model(sliced_df[column])
                    d_[column] = forecast
                for column in sliced_df.columns:
                    expected_vol_df.at[sliced_df.index.values[-1], column] = d_[column]
            expected_vol_df = expected_vol_df.shift(1)
            expected_vol_df = 1/expected_vol_df
            expected_vol_df['sum_of_stds'] = expected_vol_df.sum(axis=1)
            excluded_days = expected_vol_df.index.values[:window-1] 
            eda = expected_vol_df.index.isin(excluded_days)
            expected_vol_df = expected_vol_df.dropna()
            self.test_to_see = expected_vol_df
            weights_df = expected_vol_df.loc[:,"TLT":"SPY"].div(expected_vol_df["sum_of_stds"], axis=0)
            port_ret = weights_df.mul(self.train_data[~eda])
            port_ret['daily_returns'] = port_ret.sum(axis=1)
            port_ret['cumulative_returns'] = port_ret['daily_returns'].cumsum()
            
            d_ = {'window': window, 'total_return': port_ret['cumulative_returns'].values[-1]}
            days_result_dict = days_result_dict.append(d_, ignore_index=True)
        
        max_return = np.max(days_result_dict['total_return'])
        self.results_table = days_result_dict
        self.best_exp_period = int(days_result_dict[days_result_dict['total_return']==max_return].window.values[0])
        # print(self.best_period)
        
        best_window = self.best_exp_period
        expected_vol_df = pd.DataFrame(columns=self.train_data.columns, index=self.train_data.index)
        for row in range(0, len(expected_vol_df)):
            
            sliced_df = self.train_data.iloc[row:row+best_window]
            d_ = {}
            for column in sliced_df.columns:
                forecast = self.garch_model(sliced_df[column])
                d_[column] = forecast
            for column in sliced_df.columns:
                expected_vol_df.at[sliced_df.index.values[-1], column] = d_[column]
                
        expected_vol_df = expected_vol_df.shift(1)
        expected_vol_df = 1/expected_vol_df
        expected_vol_df['sum_of_stds'] = expected_vol_df.sum(axis=1)
        excluded_days = expected_vol_df.index.values[:window-1] 
        eda = expected_vol_df.index.isin(excluded_days)
        expected_vol_df = expected_vol_df.dropna()
        weights_df = expected_vol_df.loc[:,"TLT":"SPY"].div(expected_vol_df["sum_of_stds"], axis=0)

        port_ret = weights_df.mul(self.train_data[~eda])
        port_ret['daily_returns'] = port_ret.sum(axis=1)
        port_ret['expV_cumulative_returns'] = port_ret['daily_returns'].cumsum()
        
        train_data_for_eqw = self.train_data[~eda].copy()
        train_data_for_eqw['TLT'] = train_data_for_eqw['TLT'] * 0.35
        train_data_for_eqw['SPY'] = train_data_for_eqw['SPY'] * 0.55
        train_data_for_eqw['GLD'] = train_data_for_eqw['GLD'] * 0.1
        train_data_for_eqw['daily_returns'] = train_data_for_eqw.sum(axis=1)
        train_data_for_eqw['stdW_cumulative_returns'] = train_data_for_eqw['daily_returns'].cumsum()
        
        df_combined = train_data_for_eqw[['stdW_cumulative_returns']].join(port_ret[['expV_cumulative_returns']])
        df_combined.plot()
        plt.title("Expected Garch Vol Weighted vs Standard Weights: Training Data Set")
        plt.xticks(rotation=70)
        plt.show()
        
        sharpe_ratio_of_stdW = self.sr*np.mean(train_data_for_eqw['daily_returns'])/np.std(train_data_for_eqw['daily_returns'])
        sharpe_ratio_of_ExpV = self.sr*np.mean(port_ret['daily_returns'])/np.std(port_ret['daily_returns'])
        print("Optimized rolling window period is:", self.best_exp_period)
        print("Sharpe Ratio of Standard Weighting:", sharpe_ratio_of_stdW)
        print("Sharpe Ratio of Expected Risk Parity Weighting:", sharpe_ratio_of_ExpV)
        
        print("Cumulative returns of ExpectedVol Strategy:", df_combined.expV_cumulative_returns.values[-1])
        
    def test(self):

        df = self.test_data
        df = df.rolling(self.hist_best_period).std().shift(1)
        df = 1/df
        df['sum_of_stds'] = df.sum(axis=1)
        weights_df = df.loc[:,"TLT":"SPY"].div(df["sum_of_stds"], axis=0)
        excluded_days = df.index.values[:self.hist_best_period-1] 
        eda = df.index.isin(excluded_days)
        port_ret_h = weights_df.mul(self.test_data[~eda])
        port_ret_h['daily_returns'] = port_ret_h.sum(axis=1)
        port_ret_h['histV_cumulative_returns'] = port_ret_h['daily_returns'].cumsum()
        #---------------------------------------------------------------------------------------------
        expected_vol_df = pd.DataFrame(columns=self.test_data.columns, index=self.test_data.index)
        for row in range(0, len(expected_vol_df)):
            
            sliced_df = self.test_data.iloc[row:row+self.best_exp_period]
            d_ = {}
            for column in sliced_df.columns:
                forecast = self.garch_model(sliced_df[column])
                d_[column] = forecast
            for column in sliced_df.columns:
                expected_vol_df.at[sliced_df.index.values[-1], column] = d_[column]
                
        expected_vol_df = expected_vol_df.shift(1)
        expected_vol_df = 1/expected_vol_df
        expected_vol_df['sum_of_stds'] = expected_vol_df.sum(axis=1)
        excluded_days = expected_vol_df.index.values[:self.best_exp_period-1] 
        eda = expected_vol_df.index.isin(excluded_days)
        expected_vol_df = expected_vol_df.dropna()
        weights_df = expected_vol_df.loc[:,"TLT":"SPY"].div(expected_vol_df["sum_of_stds"], axis=0)

        port_ret_e = weights_df.mul(self.test_data[~eda])
        port_ret_e['daily_returns'] = port_ret_e.sum(axis=1)
        port_ret_e['expV_cumulative_returns'] = port_ret_e['daily_returns'].cumsum()
        #--------------------------------------------------------------------------------------------------
        test_data_for_eqw = self.test_data[~eda].copy()
        test_data_for_eqw['TLT'] = test_data_for_eqw['TLT'] * 0.35
        test_data_for_eqw['SPY'] = test_data_for_eqw['SPY'] * 0.55
        test_data_for_eqw['GLD'] = test_data_for_eqw['GLD'] * 0.1
        test_data_for_eqw['daily_returns'] = test_data_for_eqw.sum(axis=1)
        test_data_for_eqw['stdW_cumulative_returns'] = test_data_for_eqw['daily_returns'].cumsum()
        
        df_combined = test_data_for_eqw[['stdW_cumulative_returns']].join(port_ret_e[['expV_cumulative_returns']])
        df_combined = df_combined.join(port_ret_h[['histV_cumulative_returns']])
        df_combined.plot()
        plt.title("ExpectedV vs HistV vs Standard Weights: Test Data Set")
        plt.xticks(rotation=70)
        plt.show()
        
        
        sharpe_ratio_of_stdW = self.sr*np.mean(test_data_for_eqw['daily_returns'])/np.std(test_data_for_eqw['daily_returns'])
        sharpe_ratio_of_ExpV = self.sr*np.mean(port_ret_e['daily_returns'])/np.std(port_ret_e['daily_returns'])
        sharpe_ratio_of_HistV = self.sr*np.mean(port_ret_h['daily_returns'])/np.std(port_ret_h['daily_returns'])
        
        print("Sharpe Ratio of Standard Weighting:", sharpe_ratio_of_stdW)
        print("Sharpe Ratio of Expected Risk Parity Weighting:", sharpe_ratio_of_ExpV)
        print("Sharpe Ratio of Historical Risk Parity Weighting:", sharpe_ratio_of_HistV)
        print("Cumulative returns of ExpectedVol Strategy:", df_combined.expV_cumulative_returns.values[-1])
        print("Cumulative returns of HistoricalVol Strategy:", df_combined.histV_cumulative_returns.values[-1])
        print("Cumulative returns of Standard Weighting Strategy:", df_combined.stdW_cumulative_returns.values[-1])
    

if __name__ == "__main__":
    
    s = Strategy_Risk_Parity()
    s.historical_vol()
    s.expected_vol()
    s.test()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            