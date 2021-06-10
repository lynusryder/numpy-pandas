from backtest import *
trading_cost = 0.001

# Initialize
s = strategies(trading_cost)

# Momentum training
mom_param, mom_train_sr, mom_train_ret_df = s.optimize_momentum()
plot_cummulative_returns(mom_train_ret_df, 'Mom_Train_Cumret')

# Momentum test
returns_df = s.test_retdf
rank_df = s.get_rankdf(returns_df)
weight_vol_df = s.get_voldf(40, returns_df)
mom_test_sr, mom_test_ret_df, mom_test_weights = s.get_momentum_results(mom_param, returns_df, rank_df, weight_vol_df)
plot_cummulative_returns(mom_test_ret_df, 'Mom_Test_Cumret')

# HRP
hrp_test_sr, hrp_test_ret_df = s.optimize_hrp()
plot_cummulative_returns(hrp_test_ret_df, 'HRP_Test_Cumret')

combined_sr, combined_ret_df = s.equal_weighted_alphas(mom_test_weights, s.hrp_weights)
plot_cummulative_returns(combined_ret_df, 'Combined_Test_Cumret')
