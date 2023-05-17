import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tqdm import tqdm

def get_returns_data(freq = 'D'):
    
    directory = r'C:\Users\Linus Ong\Desktop\Internships & Jobs' # <------------------------- change as per needed
    filename = 'data.csv'
    file = os.path.join(directory, filename)
    df = pd.read_csv(file)

    
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    if freq:
        df = df.resample(freq).last()
    returns_df = df.pct_change().dropna()
    
    d_ = {'D': 252,
          'W': 52,
          'M': 12}
    
    return returns_df, d_[freq]

class Optimize:
    
    def __init__(self, prices):
        
        self.prices = prices[0]
        self.freq_number = prices[1]
        self.maxiter = 1000 #int(1e10)
        self.tiebreak = 'first'
        self.tolcon = 1.0e-10
        self.eps = 2e-6
        
        self.test_size = 0.2
        self.train_test_split()
        
    def train_test_split(self):
        
        self.data = self.prices.copy()
        self.train_data, self.test_data = self.data[:round((1-self.test_size)*len(self.data))], self.data[round((1-self.test_size)*len(self.data)):]
        
        
    def pf_var(self, w, V): # Calculate portfolio variance
        
        w = np.matrix(w)
        V = np.matrix(V)
        
        return (w*V*w.T)[0,0]
    
    def volatility_objective(self, x):
        
        V = self.V
        sig_p = np.sqrt(self.pf_var(x,V))
        
        return sig_p

    
    def return_objective(self, x):
        
        R = self.R
        ret = np.matrix(x)*np.matrix(R).T
        
        return -ret[0,0] # minus sign because using scipy minimize
    
            
    def _get_covmatrix_return(self, temp_df):

        corr = np.matrix(temp_df.corr())
        std = np.matrix(temp_df.std())
        self.corr = corr
        self.std = std
        V_1 = np.matmul(std, std.T) # 6x6
        V_2 = np.multiply(corr, V_1)
        self.V = V_2
        self.ExpCovariance = pd.DataFrame(V_2)

        # self.R = [0.05,0.1,0.03,0.03,0.05]
        self.R = list(temp_df.mean())
        
    def _get_constraints(self, target_vol=0.02):
        
        # is more than >
        buffer = target_vol * 0.01
        covar = self.ExpCovariance
        c = [{'type':'ineq',   'fun': lambda x: - np.sum(x) + 1.0},
             {'type':'ineq',   'fun': lambda x, target_vol=target_vol: np.sqrt(self.freq_number*x.T.dot(covar.dot(x))) - target_vol - buffer},
             {'type':'ineq',   'fun': lambda x, target_vol=target_vol: -np.sqrt(self.freq_number*x.T.dot(covar.dot(x))) + target_vol + buffer}]
        
        bounds_list = []
        inverse = 1/self.std
        sum_of_inverses = np.sum(inverse)
        for i in range(0,len(self.prices.columns)):
            max_weight = inverse[0, i]/sum_of_inverses
            tuple_to_insert = (0, max_weight)
            bounds_list.append(tuple_to_insert)
        # bounds_list = [(0,1)]*len(self.prices.columns)
        bounds = tuple(bounds_list)
        
        cons = tuple(c)
        return cons, bounds
    
    def opt(self, df, rolling_window=5, plot=False):

        weights_df = pd.DataFrame(columns = ['Date']+list(df.columns))
        
        dates = []
        for i in tqdm(range(rolling_window, len(df)-1)):
            
            dates.append(df.index[i])
            temp_df = df[i-rolling_window:i]
        
            self._get_covmatrix_return(temp_df)
            cons, bounds = self._get_constraints()
            
            options = {'disp':False,
                       'ftol': self.tolcon,
                       'eps': self.eps
                       }
            V = pd.DataFrame(self.V, columns=df.columns)
    
            w0 = np.diag(1/V)/sum(np.diag(1/V))
            
            weights = minimize(self.return_objective, 
                                w0, # starting guess
                                method = 'SLSQP', # 'SLSQP'
                                constraints = cons, # matrix of constraints
                                bounds = bounds, # matrix of bounds
                                options = options)
            
            w = np.asmatrix(weights.x)[0]
            
            list_of_weights = w.tolist()[0]
            dict_to_append = {'Date': df.index[i],
                              'TLT': list_of_weights[0],
                              'GLD': list_of_weights[1],
                              'SPY': list_of_weights[2],}
            weights_df = weights_df.append(dict_to_append, ignore_index=True)
        
        self.dates = dates
        weights_df = weights_df.set_index('Date').shift(1)

        filtered_returns = df[df.index.isin(dates)].shift(1).dropna()
        portfolio_returns = filtered_returns * weights_df
        portfolio_returns['daily_port_rets'] = portfolio_returns.sum(axis=1)
        portfolio_returns['cum_port_rets'] = portfolio_returns['daily_port_rets'].cumsum()
        
        col_list = list(df.columns)
        portfolio_returns['sum_of_weights']=weights_df[col_list].sum(axis=1)
        self.pf_rets = portfolio_returns
        
        sr = np.mean(portfolio_returns['daily_port_rets'])/np.std(portfolio_returns['daily_port_rets'])*np.sqrt(self.freq_number)
        print("Annualized Sharpe Ratio:", sr)
        print("Annualized Returns:", np.mean(portfolio_returns.daily_port_rets.values)*self.freq_number)
        cum_ret = portfolio_returns.cum_port_rets.values[-1]
        
        if plot:
            
            normal_rp_strat = self.normal_risk_parity(df, rolling_window)
            
            df_combined = portfolio_returns[['cum_port_rets']].join(normal_rp_strat[['normal_rp_cumulative_returns']])
            self.combined_returns = df_combined

            nrpsr = np.mean(normal_rp_strat['daily_returns'])/np.std(normal_rp_strat['daily_returns'])*np.sqrt(self.freq_number)

            print("Normal Risk Parity Annualized Sharpe Ratio:", nrpsr)
            print("Normal Risk Parity Annualized Returns:", np.mean(normal_rp_strat.daily_returns.values)*self.freq_number)
            df_combined.plot()
            plt.title("Standard Weights vs Risk Parity x Target Vol: Testing Data Set")
            plt.xticks(rotation=70)
            plt.show()
        
        return weights_df, portfolio_returns, sr, cum_ret
    
    def normal_risk_parity(self, df, window):
        
        returns = df.copy()
        df = df.rolling(window).std().shift(1)
        df = 1/df
        df['sum_of_stds'] = df.sum(axis=1)
        weights_df = df.loc[:,"TLT":"SPY"].div(df["sum_of_stds"], axis=0)
        excluded_days = df.index.values[:window-1] 
        eda = df.index.isin(excluded_days)
        port_ret = weights_df.mul(returns[~eda])
        port_ret['daily_returns'] = port_ret.sum(axis=1)
        port_ret['normal_rp_cumulative_returns'] = port_ret['daily_returns'].cumsum()
        
        return port_ret
        
    def opt_others(self):
        
        # a month to 4 months
        if self.freq_number == 252: #D
            x = 20
            y = 80
            step_size = 10
        elif self.freq_number == 52: #W
            x = 4
            y = 16
            step_size = 2
        elif self.freq_number == 12: #M
            x = 2
            y = 12
            step_size = 1
            
        rw_n_rets = pd.DataFrame(columns=['rolling_wind', 'cum_ret', 'sr'])
        for i in range(x, y, step_size):
            _, _, sr, cum_ret = self.opt(self.train_data, rolling_window=i)
            to_add_d = {'rolling_wind': i,
                        'cum_ret': cum_ret,
                        'sr': sr}
            rw_n_rets = rw_n_rets.append(to_add_d, ignore_index=True)
            
        max_sr = np.max(rw_n_rets['sr'])
        cum_ret = int(rw_n_rets[rw_n_rets['sr']==max_sr].cum_ret.values[0])
        rw = int(rw_n_rets[rw_n_rets['sr']==max_sr].rolling_wind.values[0])
        
        return rw, cum_ret, max_sr
    
    def test(self, freq, best_rw):
        
        d_ = {'D': 252,
              'W': 52,
              'M': 12}
        self.freq_number = d_[freq]
        return self.opt(self.test_data, rolling_window=best_rw, plot=True)
            
if __name__ == "__main__":
    
    results_df = pd.DataFrame(columns=['Freq', 'Rolling_Wind', 'SR', 'Cum_Ret'])
    backtest = True
    if backtest:
        for freq in ['W', 'M']:
            
            print("Resampling:", freq)
            returns_df = get_returns_data(freq)
            O = Optimize(returns_df)
            best_rw, max_return, best_sr = O.opt_others()
            
            to_add_d = {'Freq': freq, 
                        'Rolling_Wind': best_rw, 
                        'SR': best_sr, 
                        'Cum_Ret': max_return}
            results_df = results_df.append(to_add_d, ignore_index=True)

    best_SR = np.max(results_df.SR.values)
    optimal_freq, optimal_rw = results_df[results_df['SR']==best_SR].Freq.values[0], results_df[results_df['SR']==best_SR].Rolling_Wind.values[0] 
    test = True
    if test:
        returns_df = get_returns_data(freq)
        O = Optimize(returns_df)
        weights_df, portfolio_returns, sr, cum_ret = O.test(optimal_freq, optimal_rw)
        weights_df['Sum_Of_Weights'] = weights_df.sum(axis=1)
    
    to_save = False
    if to_save:
        directory = "C:\\Users\\Linus Ong\\Desktop\\Internships & Jobs\\"
        weights_df.to_csv(directory+"test_weights.csv")
        portfolio_returns.to_csv(directory+"test_portfolio_returns.csv")
        O.combined_returns.to_csv(directory+"combined_returns.csv")
    
    


    