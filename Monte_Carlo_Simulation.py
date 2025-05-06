import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf

# import data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockList = ['CBA','BHP','NAB','WBC','STO']
stocks = [stock + '.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=500)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)
#print(meanReturns)

def make_positive_definite(matrix, epsilon=1e-5):
    n = matrix.shape[0]
    identity = np.identity(n)
    return matrix + epsilon * identity

covMatrix = make_positive_definite(covMatrix)
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

print(weights)

# Monte Carlo method
# number of simulations

sims = 100
T = 100 # time frame in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, sims), fill_value=0.0)

initialPortfolio = 10000

for m in range(0, sims):
    # MC loops
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix) # Works out what the lower triangle is in a Cholesky decomposition
    # Cholesky decomposition requires the covariance matrix to be positive definite
    dailyReturns = meanM + np.inner(L, Z) # This is the formula from Cholesky Decomposition  Assuming daily returns are distributed by a Multivariate Normal Distribution
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio
plt.plot(portfolio_sims)
plt.ylabel('Portfolio value (£)')
plt.xlabel('Days')
plt.show()