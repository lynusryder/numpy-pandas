import os
import pandas as pd
import numpy as np
from scipy import stats
from time import strptime
from datetime import date
from scipy import special
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tqdm import tqdm

def get_returns_data():
    
    directory = r'C:\Users\Linus Ong\Desktop\Internships & Jobs' # <------------------------- change as per needed
    filename = 'Prices.xlsx'
    file = os.path.join(directory, filename)
    btc = pd.read_excel(file, sheetname='BTCUSD')
    eqty = pd.read_excel(file, sheetname='SPY')
    commods = pd.read_excel(file, sheetname='BCOM')
    bonds = pd.read_excel(file, sheetname='USBONDS')
    fx = pd.read_excel(file, sheetname='USDIndex')
    
    btc['Date'] = pd.to_datetime(btc['Date']).dt.date
    eqty['Date'] = pd.to_datetime(eqty['Date']).dt.date
    bonds['Date'] = pd.to_datetime(bonds['Date']).dt.date
    fx['Date'] = pd.to_datetime(fx['Date']).dt.date
    
    for ind in commods.index.values:
        
        date_ = commods.iloc[ind].Date
        splited_date = date_.split(", ")
        year = int(splited_date[1])
        month_day = splited_date[0].split(" ")
        month = strptime(month_day[0],'%b').tm_mon
        day = int(month_day[1])
        real_date = date(year, month, day)
        
        commods.at[ind, 'Date'] = real_date
        
    merged = pd.merge(eqty, btc, how='left', on='Date')
    merged = pd.merge(merged, bonds, how='left', on='Date')
    merged = pd.merge(merged, fx, how='left', on='Date')
    merged = pd.merge(merged, commods, how='left', on='Date')
    
    merged['Date'] = pd.to_datetime(merged['Date'])
    merged_df = merged.set_index('Date')
    returns_df = merged_df.resample('W').last().pct_change().iloc[1:]
    
    return returns_df

class Optimize:
    
    def __init__(self, prices):
        
        self.prices = prices
        
        self.maxiter = 1000 #int(1e10)
        self.tiebreak = 'first'
        self.tolcon = 1.0e-10
        self.eps = 2e-6
        
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
        
        return ret[0,0]
        
    def ValueAtRisk(self, mu, sigma, conflevel, horizon, target):
        
        logmu = horizon * (2*np.log(1+mu) - 0.5*np.log(np.power(sigma, 2) + np.power((1+mu), 2) ))
        logsigma = np.sqrt(horizon) * np.sqrt(np.log(np.power(sigma,2)/np.power((1+mu), 2) + 1))
            
        if sigma != 0:
            
            probability = 1-conflevel
            mean = logmu
            std = logsigma
            VaRout = stats.lognorm(std, scale=np.exp(mean)).ppf(probability) - np.power((1+target), horizon)
            
        elif target > mu:
            
            VaRout = np.power((1 + mu), horizon) - np.power((1 + target), horizon)
            
        else:
            
            VaRout = 0
            
        return VaRout
    
    def CVaR(self, mu, sigma, c, horizon, target, disttype): # c is optconf
        
        if disttype == 'lognormal':
            
            logmu = horizon * (2 *np.log(1+mu) - 0.5*np.log(np.power(sigma, 2) + np.power((1+mu), 2) ))
            logsigma = np.sqrt(horizon) * np.sqrt(np.log((np.power(sigma,2)/np.power((1+mu),2) + 1)))
            p = 1-c
            var = self.ValueAtRisk(mu, sigma, c, horizon, 0)

            first = 0.5
            second = np.exp(logmu + np.multiply(0.5, np.power(logsigma, 2)))
            third_1 = np.log(1+var)-logmu-np.power(logsigma, 2)
            third_2 = (np.sqrt(2)*logsigma)
            third = 1 + special.erf(np.divide(third_1, third_2))
            CVaR = np.divide(np.multiply(np.multiply(first, second), third), (p)) - (1+target)**horizon

        else:
            
            p=1-p

            middle_1 = sigma
            middle_2_1 = np.sqrt(2*math.pi)
            middle_2_2 = np.exp(np.power(special.erfinv(np.multiply(2, p) -1),2))
            middle_2_3 = (1-p)
            middle_2 = np.multiply(middle_2_1 * middle_2_2, middle_2_3)
            
            middle = np.multiply(middle_1, np.power(middle_2, -1))
            CVaR = mu + middle
            CVaR = 2*mu - CVaR
        
        return CVaR
    
    def CVaROpt(self, x):
        
        sigma, mu = self.volatility_objective(x), self.return_objective(x)
        return -self.CVaR(mu, sigma, 0.05, 1, 0, 'lognormal')
    
            
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
        
    def _get_constraints(self, target_return=None, R=None):
        
    
        c = [{'type':'eq',   'fun': lambda x: np.sum(x)-1.0}]

        bounds_list = [(0,1),(0,1),(0,1),(0,1),(0,1)]
        bounds = tuple(bounds_list)
        
        cons = tuple(c)
        return cons, bounds
    
    def opt(self, rolling_window=11, plot=False):
        
        index_values = [100]
        weights_df = pd.DataFrame(columns = self.prices.columns)
        returns_list = []
        
        dates = []
        for i in tqdm(range(rolling_window, len(self.prices)-1)):
            
            dates.append(self.prices.index[i])
            temp_df = self.prices[i-rolling_window:i]
        
            self._get_covmatrix_return(temp_df)
            cons, bounds = self._get_constraints()
            
            options = {'disp':False,
                       'ftol': self.tolcon,
                       'eps': self.eps
                       }
            V = pd.DataFrame(self.V, columns=self.prices.columns)
    
            w0 = np.diag(1/V)/sum(np.diag(1/V))
            
            weights = minimize(self.CVaROpt, 
                                w0, # starting guess
                                method = 'SLSQP', # 'SLSQP'
                                constraints = cons, # matrix of constraints
                                bounds = bounds, # matrix of bounds
                                options = options)
            
            w = np.asmatrix(weights.x)[0]
            returns = list(self.prices.iloc[i+1])
            ret = w*np.matrix(returns).T
            final_return = ret[0,0]
            returns_list.append(final_return)
            index_values.append(index_values[-1]+100*final_return)
            
            list_of_weights = w.tolist()[0]
            dict_to_append = {'eqty': list_of_weights[0],
                              'btc': list_of_weights[1],
                              'bonds': list_of_weights[2],
                              'fx': list_of_weights[3],
                              'commods': list_of_weights[4],}
            weights_df = weights_df.append(dict_to_append, ignore_index=True)
        
        if plot:

            ax = plt.gca()
            ax.plot(dates, index_values[1:])
            
            plt.ylabel('Index Values') 
            plt.xlabel('Date') 
            plt.title("Portfolio Performance") 
            plt.show() 
        
        return weights_df, returns_list
    
    def opt_others(self):
        
        rolling_window_dict = {}
        for i in range(5, 15):
            _, returns_list = self.opt(i)
            rolling_window_dict[i] = np.sum(returns_list)
            
        optimal_rolling_window = max(rolling_window_dict, key=rolling_window_dict.get)
        print("Optimal rolling_window:", optimal_rolling_window)
        
        weights_df, returns_list = self.opt(rolling_window=optimal_rolling_window, plot=True)
        
        return weights_df, returns_list
            
if __name__ == "__main__":
    
    returns_df = get_returns_data()
    O = Optimize(returns_df)
    weights_df, returns_list = O.opt(plot=True)
    weights_df.plot()

    