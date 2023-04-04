import yfinance as yf
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Strategy_130_30:
    
    def __init__(self):
        
        self.directory = "C:\\Users\\Linus Ong\\Downloads"
        
        self.start_date = '2018-01-01'
        self.end_date = '2023-02-28'
        self.freq = '1d'
        
        self.test_size = 0.2
        
        try:
            data = pd.read_csv("C:\\Users\\Linus Ong\\Desktop\\Internships & Jobs\\rpt_data.csv")
            data.set_index('Date', inplace=True)
            self.data = data.pct_change().dropna()
        except:
            self.get_long_stocks()
            self.get_short_stocks()
            self.get_data()
            
        self.further_cleaning_data()
        self.train_test_split()
        
    def get_long_stocks(self):
        
        filename = "2023 Call Long with EPS.xlsx"
        file_dir = os.path.join(self.directory, filename)
        df = pd.read_excel(file_dir)
        
        self.long_stocks = list(x for x in df.Company.values if str(x) != 'nan')
        
    def get_short_stocks(self):
        
        filename = "2022 short (1).xlsx"
        file_dir = os.path.join(self.directory, filename)
        df = pd.read_excel(file_dir)
        
        self.short_stocks = list(x for x in df.Company.values if str(x) != 'nan')
        
    def save_data(self, df):
        
        df.to_csv("C:\\Users\\Linus Ong\\Desktop\\Internships & Jobs\\rpt_data.csv")

    def get_data(self):
        
        stocks = self.long_stocks + self.short_stocks
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
        try:
            self.save_data(data)
            print("Data saved.")
        except:
            print("Error saving data.")
            
    def further_cleaning_data(self):
        
        temp_data = self.data.copy()
        mask = (temp_data == 0).sum()/len(temp_data) < 0.5
        new_data = temp_data[temp_data.columns[mask]]
        self.data = new_data
            
    def train_test_split(self):
        
        self.train_data, self.test_data = self.data[:round(0.8*len(self.data))], self.data[round(0.8*len(self.data)):]
        
    def sharpe_ratio(self, y):
        
        return np.sqrt(126) * (y.mean() / y.std())
        
    def sharpe_ratio_ranking_opt(self): 
        
        days_result_dict = pd.DataFrame(columns = ['day', 'total_return'])
        for day in tqdm(range(14, 42)):
            
            rolling_days = day
            
            df = self.train_data
            df = df.rolling(rolling_days).apply(self.sharpe_ratio).shift(1)
            excluded_days = df.index.values[:rolling_days-1] 
            eda = df.index.isin(excluded_days)
            rank_df = df[~eda].rank(axis=1)
            
            weights_of_bot_15 = -0.02 # 30 stocks
            weights_of_mid = (100/(len(df.columns)-30))/100
            weights_of_up_15 = 0.02 # 30 stocks
            
            weights_df = np.where(rank_df < 16, weights_of_bot_15, rank_df)
            weights_df = np.where(weights_df > len(df.columns)-15, weights_of_up_15, weights_df)
            weights_df = np.where((weights_df <= len(df.columns)-15)&(weights_df >=16), weights_of_mid, weights_df)
            weights_df = pd.DataFrame(weights_df, columns=df.columns, index=rank_df.index)
            
            port_ret = weights_df.mul(self.train_data[~eda])
            port_ret['daily_returns'] = port_ret.sum(axis=1)
            port_ret['130_30_cumulative_returns'] = port_ret['daily_returns'].cumsum()
    #        port_ret['cumulative_returns'].plot()
    #        plt.show()
            d_ = {'day': rolling_days, 'total_return': port_ret['130_30_cumulative_returns'].values[-1]}
            days_result_dict = days_result_dict.append(d_, ignore_index=True)
        
        
        eq_w = (100/len(df.columns))/100
        eq_w_df = np.where(rank_df>0, eq_w, eq_w)
        eq_w_df1 = pd.DataFrame(eq_w_df, columns=df.columns, index=rank_df.index)
        eq_w_port_ret = eq_w_df1.mul(self.train_data[~eda])
        eq_w_port_ret['daily_returns'] = eq_w_port_ret.sum(axis=1)
        eq_w_port_ret['eqw_cumulative_returns'] = eq_w_port_ret['daily_returns'].cumsum()
        
#        df_combined = eq_w_port_ret[['eqw_cumulative_returns']].join(port_ret[['130_30_cumulative_returns']])
#        df_combined.plot()
#        plt.title("130/30 Equity Strategy: Training Data Set")
#        plt.xticks(rotation=70)
#        plt.show()
        
        max_return = np.max(days_result_dict['total_return'])
        self.results_table = days_result_dict
        self.best_period = int(days_result_dict[days_result_dict['total_return']==max_return].day.values[0])
    
    def test(self):
        
        df = self.test_data
        df = df.rolling(self.best_period).apply(self.sharpe_ratio).shift(1)
        excluded_days = df.index.values[:self.best_period-1] 
        eda = df.index.isin(excluded_days)
        rank_df = df[~eda].rank(axis=1)
        
        weights_of_bot_15 = -0.02 # 30 stocks
        weights_of_mid = (100/(len(df.columns)-30))/100
        weights_of_up_15 = 0.02 # 30 stocks
        
        weights_df = np.where(rank_df < 16, weights_of_bot_15, rank_df)
        weights_df = np.where(weights_df > len(df.columns)-15, weights_of_up_15, weights_df)
        weights_df = np.where((weights_df <= len(df.columns)-15)&(weights_df >=16), weights_of_mid, weights_df)
        weights_df = pd.DataFrame(weights_df, columns=df.columns, index=rank_df.index)
        self.weights_df = weights_df
        
        port_ret = weights_df.mul(self.test_data[~eda])
        port_ret['daily_returns'] = port_ret.sum(axis=1)
        port_ret['130_30_cumulative_returns'] = port_ret['daily_returns'].cumsum()
        self.port_returns = port_ret
        
        
        eq_w = (100/len(df.columns))/100
        eq_w_df = np.where(rank_df>0, eq_w, eq_w)
        eq_w_df1 = pd.DataFrame(eq_w_df, columns=df.columns, index=rank_df.index)
        eq_w_port_ret = eq_w_df1.mul(self.test_data[~eda])
        eq_w_port_ret['daily_returns'] = eq_w_port_ret.sum(axis=1)
        eq_w_port_ret['eqw_cumulative_returns'] = eq_w_port_ret['daily_returns'].cumsum()
        
        df_combined = eq_w_port_ret[['eqw_cumulative_returns']].join(port_ret[['130_30_cumulative_returns']])
        df_combined.plot()
        plt.title("130/30 Equity Strategy: Testing Data Set")
        plt.xticks(rotation=70)
        plt.show()
        print("Optimized rolling period for sharpe ratio ranking:", self.best_period)
        sr = np.sqrt(252)*np.mean(port_ret['daily_returns'])/np.std(port_ret['daily_returns'])
        print("Annualized Sharpe Ratio for 130/30:", sr)

        sr = np.sqrt(252)*np.mean(eq_w_port_ret['daily_returns'])/np.std(eq_w_port_ret['daily_returns'])
        print("Annualized Sharpe Ratio for EQW:", sr)
        
        wealth_index = 1000*(1+port_ret['daily_returns']).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdown = (wealth_index - previous_peaks)/previous_peaks
        mdd = drawdown.min()*100
        print("Max drawdown for 130/30:", mdd, "%")
        wealth_index = 1000*(1+eq_w_port_ret['daily_returns']).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdown = (wealth_index - previous_peaks)/previous_peaks
        mdd = drawdown.min()*100
        print("Max drawdown for EQW:", mdd, "%")

                     
if __name__ == "__main__":
    
    s = Strategy_130_30()
    s.sharpe_ratio_ranking_opt()            
    s.test()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            