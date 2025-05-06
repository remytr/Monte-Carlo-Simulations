import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf


# Check if tickers are valid and have data
def validate_tickers(stockList):
    valid_tickers = []
    invalid_tickers = []

    for stock in stockList:
        ticker = stock + '.AX'
        data = yf.download(ticker, period='1d', progress=False)

        if data.empty or data.isnull().values.any():
            print(f"Warning: Ticker {ticker} returned no data or contains NaN values.")
            invalid_tickers.append(stock)
        else:
            print(f"Ticker {ticker} is valid.")
            valid_tickers.append(stock)

    return valid_tickers, invalid_tickers


# import data with validation
def get_data(stocks, start, end):
    print(f"\nDownloading data for {stocks}...")
    stockData = yf.download(stocks, start=start, end=end, progress=False)

    if len(stocks) == 1:
        # Handle single ticker case differently
        stockData = pd.DataFrame(stockData['Close'])
        stockData.columns = [stocks[0]]
    else:
        stockData = stockData['Close']

    # Check for any missing data
    missing_data = stockData.isnull().sum()
    if missing_data.any():
        print("Warning: Missing data detected:")
        print(missing_data[missing_data > 0])

    # Fill or drop NaN values
    print(f"Original data shape: {stockData.shape}")
    # Forward fill then backfill to handle missing values
    stockData = stockData.fillna(method='ffill').fillna(method='bfill')

    # Calculate returns
    returns = stockData.pct_change().dropna()
    print(f"Return data shape after NaN handling: {returns.shape}")

    # Display data summary
    print("\nData summary:")
    print(returns.describe())

    meanReturns = returns.mean()
    covMatrix = returns.cov()

    return meanReturns, covMatrix, returns, stockData


# List of Australian stocks
stockList = ['CBA', 'BHP', 'NAB', 'WBC', 'STO']

# Validate the tickers first
print("Validating tickers...")
valid_stocks, invalid_stocks = validate_tickers(stockList)

if not valid_stocks:
    print("No valid tickers found. Please provide valid stock symbols.")
    exit()
elif invalid_stocks:
    print(f"\nProceeding with valid tickers only: {valid_stocks}")
    print(f"Skipping invalid tickers: {invalid_stocks}")

# Add .AX extension for ASX stocks
stocks = [stock + '.AX' for stock in valid_stocks]

# Dates - Get more history to ensure sufficient data
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=800)  # More data helps reduce NaN impact

# Get the data
meanReturns, covMatrix, returns, stockData = get_data(stocks, startDate, endDate)

# Check covariance matrix properties
print("\nCovariance matrix check:")
# Is the matrix symmetric?
is_symmetric = np.allclose(covMatrix, covMatrix.T)
print(f"Matrix is symmetric: {is_symmetric}")

# Check eigenvalues to see if it's positive definite
eigenvalues = np.linalg.eigvals(covMatrix)
is_positive_definite = np.all(eigenvalues > 0)
print(f"Matrix is positive definite: {is_positive_definite}")
if not is_positive_definite:
    print(f"Smallest eigenvalue: {np.min(eigenvalues)}")


# Fix for non-positive definite matrix
def make_positive_definite(matrix, epsilon=1e-5):
    """
    Adds a small value to the diagonal of the matrix to make it positive definite
    """
    n = matrix.shape[0]
    identity = np.identity(n)
    # Add a small multiple of the identity matrix
    adjusted_matrix = matrix + epsilon * identity

    # Verify it worked
    eigenvalues = np.linalg.eigvals(adjusted_matrix)
    is_positive_definite = np.all(eigenvalues > 0)
    print(f"Adjusted matrix is positive definite: {is_positive_definite}")
    print(f"Smallest eigenvalue after adjustment: {np.min(eigenvalues)}")

    return adjusted_matrix


# Apply the fix if needed
if not is_positive_definite:
    print("\nApplying positive definite fix...")
    covMatrix = make_positive_definite(covMatrix)

# Generate random weights for the portfolio
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

print("\nPortfolio Weights:")
for stock, weight in zip(valid_stocks, weights):
    print(f"{stock}: {weight:.4f}")

# Monte Carlo method
sims = 100
T = 100  # time frame in days

# Create a matrix of mean returns (assets x days)
meanM = np.full(shape=(len(weights), T), fill_value=meanReturns.values.reshape(-1, 1))

portfolio_sims = np.full(shape=(T, sims), fill_value=0.0)
initialPortfolio = 10000

# Try Cholesky method
try:
    print("\nRunning Monte Carlo simulation with Cholesky decomposition...")
    for m in range(sims):
        Z = np.random.normal(size=(len(weights), T))
        L = np.linalg.cholesky(covMatrix)
        dailyReturns = meanM + L @ Z
        portfolio_sims[:, m] = np.cumprod(np.sum(weights * dailyReturns, axis=0) + 1) * initialPortfolio
except np.linalg.LinAlgError as e:
    print(f"Error with Cholesky: {e}")
    print("Falling back to multivariate normal method...")

    # Fallback method
    for m in range(sims):
        # Generate correlated returns directly
        daily_returns = np.random.multivariate_normal(
            meanReturns.values,
            covMatrix.values,
            T
        )

        # Calculate portfolio value over time
        portfolio_value = np.zeros(T)
        portfolio_value[0] = initialPortfolio

        for t in range(1, T):
            # Calculate portfolio return as weighted sum of asset returns
            port_return = np.sum(weights * daily_returns[t])
            portfolio_value[t] = portfolio_value[t - 1] * (1 + port_return)

        portfolio_sims[:, m] = portfolio_value

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(portfolio_sims)

# Add statistics and reference lines
final_values = portfolio_sims[-1, :]
median_final = np.median(final_values)
percentile_5 = np.percentile(final_values, 5)
percentile_95 = np.percentile(final_values, 95)

plt.axhline(y=median_final, color='r', linestyle='--',
            label=f'Median: ${median_final:.2f}')
plt.axhline(y=percentile_5, color='orange', linestyle='--',
            label=f'5th Percentile: ${percentile_5:.2f}')
plt.axhline(y=percentile_95, color='green', linestyle='--',
            label=f'95th Percentile: ${percentile_95:.2f}')
plt.axhline(y=initialPortfolio, color='black', linestyle='-',
            label=f'Initial: ${initialPortfolio}')

plt.title('Monte Carlo Simulation of Portfolio Value')
plt.xlabel('Days')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print summary statistics
print("\nPortfolio Statistics after", T, "days:")
print(f"Starting value: ${initialPortfolio}")
print(f"Mean final value: ${np.mean(final_values):.2f}")
print(f"Median final value: ${median_final:.2f}")
print(f"Minimum final value: ${np.min(final_values):.2f}")
print(f"Maximum final value: ${np.max(final_values):.2f}")
print(f"Standard deviation: ${np.std(final_values):.2f}")
print(f"5th percentile: ${percentile_5:.2f}")
print(f"95th percentile: ${percentile_95:.2f}")
print(f"Probability of gain: {np.mean(final_values > initialPortfolio):.2%}")