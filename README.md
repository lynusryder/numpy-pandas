# numpy-pandas

**Description of proposed trading strategy**

Training period: 1988-01-07 to 2011-10-27
Testing period: 2011-10-28 to 2017-10-10

Strategy 1: Cross-sectional Momentum
We first get the returns data, split it into train and test sets, and then filter out stocks with high 
volatility from the training set. From the one-month returns data, we rank every asset on a daily 
basis based on their returns and we long the top x and short the bottom x assets. This is based on 
the assumption that assets that have had a higher (lower) one-month return would continue to have 
a higher (lower) return in subsequent time periods. We then bet on this relative spread by taking 
long positions in those that have exhibited strong returns over the past one month and short 
positions in those that have exhibited lower returns over the past one month. The parameter 
optimized here is the number of assets to take long and short positions in at any one point of time.

Strategy 2: Hierarchical Risk Parity
This optimization method outputs weights based on the correlation of assets and their distance 
matrix. First, a distance matrix is formed based on pair wise correlations among the portfolio of 
assets. This distance matrix then clusters the assets into a tree, the same way hierarchical clustering 
clusters elements. Within each branch of the tree, the minimum variance portfolio is calculated. By 
iterating over each branch or level, the algorithm combines the sub portfolios at each branch and 
outputs an optimal weight for each asset.

Strategy 3: Combined alphas
The final strategy is a combination of strategy 1 and 2 with equal weights to see if the sharpe ratio 
can be enhanced.

**Performance statistics**

Strategy 1: Cross-sectional Momentum
Parameter chosen: Top/bottom 9
Annualized Sharpe Ratio:
Momentum Training Set 0.143 
Momentum Test Set 0.529

Strategy 2: Hierarchical Risk Parity
Output weights from optimizing the training set were used on the test set throughout the testing period.
Annualized Sharpe Ratio: HRP Test Set 0.852

Strategy 3: Equally weighted signals for strategy 1 and 2
Annualized Sharpe Ratio: Combined Test Set 0.540
