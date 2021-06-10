import operator
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from pypfopt import hierarchical_portfolio

def plot_cummulative_returns(daily_ret_df, title):
    series = daily_ret_df.iloc[::-1]
    cumret = series.cumsum()+1
    fig = plt.figure()
    ax = plt.axes()
    y = list(cumret.values)
    x = list(cumret.index)
    # Create bars and choose color
    plt.plot(x, y)
     
    # Add title and axis names
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cummulative Returns')
    plt.show()
    
class strategies():
    def __init__(self, trading_cost):
        filepath = r'C:\Users\Linus Ong\Desktop\Internships & Jobs\FSI\Data.xlsx'
        df = pd.read_excel(filepath)
        self.trading_cost = trading_cost
        
        # Manipulate df
        
        df = df.rename(columns={'Unnamed: 0': 'Date'})
        df = df.set_index('Date')
        returns_df = df.diff(periods=-1)
        returns_df = returns_df.dropna()
        
        length = len(returns_df)
        test_length = int(round(0.2*length,0))
        train_df = returns_df.copy()[test_length:]
        
        # dropping assets with high volatility in the training set
        dropped = []
        for column in train_df:
            vol = train_df[column].std()
            if vol > 2:
                train_df = train_df.drop(columns=[column])
                dropped.append(column)
                
        self.train_retdf = train_df
        self.test_retdf = returns_df[:test_length].drop(columns=dropped)
        self.returns_df = returns_df
        
    #############################
    # Df manipulation functions #
    #############################
        
    def get_rankdf(self, df):
        result = df.rank(axis=1, method='first')
        return result
    
    def get_voldf(self, window_size, df):
        rolling = df.iloc[::-1].rolling(window_size).std().iloc[::-1]
        return rolling
    
    ############################
    # First Strategy: Momentum #
    ############################
    
    def get_momentum_results(self, top_bot_no, returns_df, rank_df, weight_vol_df):
        
        signal_df = rank_df.copy()
        bottom = top_bot_no
        top = len(rank_df.columns) - top_bot_no +1
        signal_df[signal_df<=bottom] = -1 # short bottom i
        signal_df[signal_df>=top] = 1 # long bottom i
        signal_df[(signal_df<=top)&(signal_df>=bottom)] = 0
        signal_df = signal_df * (1-self.trading_cost)
        weights_df = signal_df * weight_vol_df # assign vol weighted weights
        weights_df = weights_df.shift(-1, axis=0) # position taken next day
        sig_x_ret_df = returns_df * weights_df
        daily_ret_df = sig_x_ret_df.sum(axis=1)
        average_ret = daily_ret_df.mean(skipna=True)
        average_std = daily_ret_df.std(skipna=True)
        sr = average_ret/average_std*np.sqrt(252) # sharpe ratio
        return sr, daily_ret_df, weights_df
        
    def optimize_momentum(self):
        
        # rank 1 month returns
        returns_df = self.train_retdf.copy()
        cummulative_ret_period = 20
        onem_cumret_df = returns_df.iloc[::-1].rolling(cummulative_ret_period).sum().iloc[::-1]
        rank_df = self.get_rankdf(onem_cumret_df)
        
        # vol dataframe for assigning weights 
        vol_df = self.get_voldf(40, returns_df)
        weight_vol_df = 0.1/vol_df # higher vols get lower weights
        weight_vol_df = weight_vol_df.replace(-np.inf, np.nan)
        weight_vol_df = weight_vol_df.replace(np.inf, np.nan)
        
        # long top x, short bottom x
        param_dict = dict()
        x = [6,9,13]
        for i in x:
            sharpe_ratio, daily_ret_df, w = self.get_momentum_results(i, returns_df, rank_df, weight_vol_df)
            param_dict[i] = [sharpe_ratio, daily_ret_df]
        
        param = max(param_dict.items(), key=operator.itemgetter(1))[0]
        best_sr = param_dict[param][0]
        best_daily_ret_df = param_dict[param][1]
        
        return param, best_sr, best_daily_ret_df
    
    #############################################
    # Second Strategy: Hierarchical Risk Parity #
    #############################################
    
    def optimize_hrp(self):
        
        train_retdf = self.train_retdf.copy()
        cov_matrix = train_retdf.cov()
        h = hierarchical_portfolio.HRPOpt(returns=train_retdf, cov_matrix=cov_matrix)
        d = h.optimize()
        exp_ret, exp_vol, exp_sr = h.portfolio_performance()
        test_retdf = self.test_retdf.copy()
        
        signals_df = pd.DataFrame(0, columns=test_retdf.columns, index=test_retdf.index)
        signals_df = signals_df.apply(lambda x: x+d[x.name])
        weights_df = signals_df.shift(-1, axis=0) * (1-self.trading_cost)# position taken next day
        self.hrp_weights = weights_df
        sig_x_ret_df = test_retdf * weights_df
        daily_ret_df = sig_x_ret_df.sum(axis=1)
        average_ret = daily_ret_df.mean(skipna=True)
        average_std = daily_ret_df.std(skipna=True)
        sr = average_ret/average_std*np.sqrt(252) # sharpe ratio
        return sr, daily_ret_df
    
    def equal_weighted_alphas(self, test_mom_alpha, test_hrp_alpha):
        test_retdf = self.test_retdf.copy()
        weights_df = 0.5 * test_mom_alpha + 0.5 * test_hrp_alpha
        
        sig_x_ret_df = test_retdf * weights_df
        daily_ret_df = sig_x_ret_df.sum(axis=1)
        average_ret = daily_ret_df.mean(skipna=True)
        average_std = daily_ret_df.std(skipna=True)
        sr = average_ret/average_std*np.sqrt(252) # sharpe ratio
        return sr, daily_ret_df
        
        
        
            
        
            
            
            
        
        
        
        
        
        
    
        
        
        