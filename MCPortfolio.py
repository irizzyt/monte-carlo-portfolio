import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- 1. PARAMETERS ---
tickers = ["NVDA", "TSM", "CRBP", "PLTR", "CRWD"]
start_date = "2023-01-01"
end_date = "2025-01-01"
risk_free_rate = 0.0322  # Annualized 3.22%

# --- 2. DATA GATHERING & FINANCIAL CALCULATIONS ---
adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    adj_close_df[ticker] = data['Close'] # Use Adj Close for accuracy

log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

mean_returns = log_returns.mean() * 252
cov_matrix = log_returns.cov() * 252
num_assets = len(tickers)

# --- 3. PORTFOLIO PERFORMANCE FUNCTIONS ---
def portfolio_performance(weights):
    ret = np.sum(mean_returns * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe

def neg_sharpe_ratio(weights):
    # We negate the Sharpe Ratio because the optimizer can only minimize functions
    return -portfolio_performance(weights)[2]

def portfolio_volatility(weights):
    return portfolio_performance(weights)[1]

# --- 4. OPTIMIZATION SETUP (CONSTRAINTS & INITIAL GUESS) ---
bounds = tuple((0, 0.3) for _ in range(num_assets)) # Max 30% in any single asset
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1} # Weights must sum to 1
initial_guess = np.array(num_assets * [1. / num_assets])

# --- 5. RUN PRIMARY OPTIMIZATIONS (Max Sharpe & Min Volatility) ---
# This block MUST come before the Efficient Frontier calculation
opt_sharpe = minimize(neg_sharpe_ratio,
                      initial_guess,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

opt_vol = minimize(portfolio_volatility,
                   initial_guess,
                   method='SLSQP',
                   bounds=bounds,
                   constraints=constraints)

# Extract the results from the optimizations
max_sharpe_weights = opt_sharpe.x
min_vol_weights = opt_vol.x
max_ret, max_vol, max_sharpe = portfolio_performance(max_sharpe_weights)
min_ret, min_vol, min_sharpe = portfolio_performance(min_vol_weights)


# --- 6. CALCULATE THE EFFICIENT FRONTIER CURVE ---
# Now that we have min_ret and max_ret, we can build the frontier
target_returns = np.linspace(min_ret, max_ret + 0.05, 100)
frontier_volatilities = []

for target_return in target_returns:
    return_constraint = {'type': 'eq', 'fun': lambda w: portfolio_performance(w)[0] - target_return}
    combined_constraints = [constraints, return_constraint]

    frontier_opt = minimize(portfolio_volatility,
                            initial_guess,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=combined_constraints)

    frontier_volatilities.append(frontier_opt.fun)

# --- 7. MONTE CARLO SIMULATION (for visualization) ---
num_portfolios = 20000
results = np.zeros((num_portfolios, 3))
for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    ret, vol, sharpe = portfolio_performance(weights)
    results[i, :] = [ret, vol, sharpe]

results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe'])

# --- 8. DISPLAY RESULTS & PLOT ---
def display_portfolio(title, weights, ret, vol, sharpe):
    print(f"\n--- {title} ---")
    print(pd.Series(weights, index=tickers, name="Weight").to_string(float_format="{:.2%}".format))
    print(f"\nExpected Annual Return: {ret:.2%}")
    print(f"Annual Volatility (Risk): {vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")

display_portfolio("Maximum Sharpe Ratio Portfolio", max_sharpe_weights, max_ret, max_vol, max_sharpe)
display_portfolio("Minimum Volatility Portfolio", min_vol_weights, min_ret, min_vol, min_sharpe)

# Plotting
plt.figure(figsize=(12, 8))
# Plot the Monte Carlo random portfolios (the "cloud")
plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'], cmap='viridis', s=15, alpha=0.6)
plt.colorbar(label='Sharpe Ratio')

# Plot the calculated Efficient Frontier curve
plt.plot(frontier_volatilities, target_returns, 'r--', linewidth=2.5, label='Efficient Frontier')

# Highlight the two optimized portfolios
plt.scatter(max_vol, max_ret, c='red', s=150, edgecolors='black', label='Max Sharpe Portfolio')
plt.scatter(min_vol, min_ret, c='blue', s=150, edgecolors='black', label='Min Volatility Portfolio')

plt.title('Efficient Frontier with Optimal Portfolios')
plt.xlabel('Annual Volatility (Risk)')
plt.ylabel('Annual Expected Return')
plt.legend()
plt.grid(True)
plt.show()