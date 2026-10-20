import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

path = r"C:\Users\npgps\OneDrive\Courses\Financial modelling\code\Project\mcs\basics\merton jump square rool  diffusion model\netflix.csv"

df = pd.read_csv(path)

df = df.rename(columns={'Close/Last':'Close'})

column_to_be_dropped = [
    'Open',
    'High',
    'Low',
    'Volume'
]

df = df.drop(columns=column_to_be_dropped)

df['Close'] = pd.to_numeric(
    df['Close']
    .astype(str)
    .replace('[$]','',regex=True),
    errors='coerce'
)

df['Date'] = pd.to_datetime(
    df['Date'],
    format='%m/%d/%Y',
)

df = df.set_index('Date')
df = df.sort_index()

# calculate log return
df['Log close price'] = np.log(df['Close'])
df['Log return'] = (df['Log close price'] -
                    df['Log close price'].shift(1))

df['Rolling volatility'] = (
    df['Log return'].rolling(window=20).std() * np.sqrt(252)
)
df['Rolling variance'] = df['Rolling volatility'] ** 2

# Heston model
dt = 1 / 252

df['Previous variance'] = df['Rolling variance'].shift(1)
df['Variance change'] = (df['Rolling variance']
                         - df['Previous variance'])
df = df.dropna()

# calculate the slope
x = sm.add_constant(df['Previous variance'])

model = sm.OLS(df['Variance change'], x).fit()
intercept = model.params['const']
slope = model.params['Previous variance']

# mean-reversion speed
kappa = - slope / dt

if kappa > 0:
    theta = intercept / (kappa * dt)
else:
    kappa = 2
    theta = df['Rolling variance'].mean()

theta = max(theta, 0.000001)

# volatility of variance
residual = model.resid
variance_shock = residual / np.sqrt(
    np.maximum(df['Previous variance'] , 0.000001) * dt
)
# xi: volatility of variance
xi = variance_shock.std()
xi = max(xi, 0.000001)

rho = np.corrcoef(
    df['Log return'],
    variance_shock
)[0,1]

mu = (
    df['Log return'].mean() * 252 + 0.5 * np.mean(
        df['Previous variance'])
)

v0 = df['Rolling variance'].iloc[-1]

# simulate
number_of_step = 252
date = pd.date_range(
    start = df.index[-1],
    periods = number_of_step+1
)
length_of_date = len(date)
number_of_path = 50000
initial_price = df['Close'].iloc[-1]

stock_path = np.zeros(
    (
        length_of_date,
        number_of_path
    )
)

variance_path = stock_path.copy()

stock_path[0] = initial_price
variance_path[0] = v0

mu = (
    df['Log return'].mean() * 252
    + 0.5 * np.mean(df['Rolling variance'])
)

stock_list = []
variance_list = []

for t in range(1, number_of_step + 1):
    previous_price = stock_path[t-1]
    previous_variance = np.maximum(
        variance_path[t - 1],
        0
    )

    stock_z = np.random.normal(
        0,
        1,
        number_of_path
    )

    independent_z = np.random.normal(
        0,
        1,
        number_of_path
    )

    variance_z = (
        rho * stock_z + np.sqrt(1 - rho ** 2) * independent_z
    )

    variance_path[t] = (previous_variance
    + kappa * (theta - previous_variance) * dt
    + xi * np.sqrt(previous_variance) * np.sqrt(dt)
                        * variance_z)

    stock_path[t] = (
        previous_price
        * np.exp(
        (mu - 0.5 * previous_variance) * dt
        + np.sqrt(previous_variance * dt) * stock_z)
    )

    stock_list.append(stock_path[t])
    variance_list.append(variance_path[t])

stock_df = pd.DataFrame(
    stock_path,
    index=date
)
variance_df = pd.DataFrame(variance_list)

plt.plot(
    stock_df.loc[
        stock_df.index.to_series().between(
            '2026-10-01', '2026-10-25')]
)
plt.title('Netflix Heston Monte Carlo Price Paths')
plt.xlabel('Trading day')
plt.ylabel('Simulated stock price')
plt.tight_layout()
plt.show()

selected_price = '2026-10-25'
print(f'\nSimulated price on {selected_price}')
print(stock_df.loc[selected_price])

# =====================================
# Investment decision analysis
# =====================================

investment_horizon = number_of_step
terminal_prices = stock_df.iloc[-1]

target_return = 0.10
stop_loss_return = -0.15

target_price = initial_price * (1 + target_return)
stop_loss_price = initial_price * (1 + stop_loss_return)

terminal_returns = terminal_prices / initial_price - 1

probability_of_profit = np.mean(terminal_prices > initial_price)

probability_of_target = np.mean(
    terminal_prices >= target_price
)

probability_of_stop_loss = np.mean(
    terminal_prices <= stop_loss_price
)

# Distribution statistics
mean_terminal_price = terminal_prices.mean()
median_terminal_price = terminal_prices.median()

percentile_5_price = terminal_prices.quantile(0.05)
percentile_95_price = terminal_prices.quantile(0.95)

expected_return = terminal_returns.mean()
median_return = terminal_returns.median()

# 95% Value at Risk as a percentage loss
var_95_return = terminal_returns.quantile(0.05)
var_95_loss = max(-var_95_return, 0)

# Expected Shortfall:
# average return among the worst 5% of outcomes
worst_5_percent = terminal_returns[
    terminal_returns <= var_95_return
]

expected_shortfall_return = worst_5_percent.mean()

# from beginning to 2027-06-08
print("\nInvestment decision analysis")
print("---------------------------------------")
print(f"Current price: ${initial_price:.2f}")
print(f"Mean terminal price: ${mean_terminal_price:.2f}")
print(f"Median terminal price: ${median_terminal_price:.2f}")
print(f"5th percentile price: ${percentile_5_price:.2f}")
print(f"95th percentile price: ${percentile_95_price:.2f}")

print(f"\nExpected return: {expected_return:.2%}")
print(f"Median return: {median_return:.2%}")

print(f"\nProbability of profit: "
      f"{probability_of_profit:.2%}")

print(f"Probability of reaching {target_return:.0%} return: "
      f"{probability_of_target:.2%}")

print(f"Probability of losing at least "
      f"{abs(stop_loss_return):.0%}: "
      f"{probability_of_stop_loss:.2%}")

print(f"\n95% Value at Risk: {var_95_loss:.2%}")
print(f"Average return in worst 5% of outcomes: "
      f"{expected_shortfall_return:.2%}")

# 33rd percentile ('2026-10-01' to '2026-10-25)

# Express returns in percentage points


new = stock_df.loc[
   (stock_df.index >= '2026-10-02') & (stock_df.index<='2026-10-25')
]

new_terminal_price = new.iloc[-1]
new_initial_price = new.iloc[0]

prob_profit = np.mean(new_terminal_price >new_initial_price)
print(f'prob_profit:{prob_profit:.3%}')



new_return = new_terminal_price / new_initial_price - 1
new_return_avg = new_return.mean()
new_return_pct = new_return * 100

# 95VaR
var_95 = new_return.quantile(0.05)
shortfall = new_return_pct[new_return_pct < var_95 *100].mean()
print(f'Expected shortfall: {shortfall:.03f}%')


# 33rd and 67th percentiles
return_33 = new_return_pct.quantile(1 / 3)
return_67 = new_return_pct.quantile(2 / 3)

plt.hist(
    new_return_pct,
    bins=50,
    edgecolor='black'
)

plt.axvline(
    return_33,
    label=f'33.33rd percentile: {return_33:.2f}%',
    color='red',
    linestyle='--'
)

plt.axvline(
    return_67,
    label=f'66.67th percentile: {return_67:.2f}%',
    color='purple',
    linestyle='--'
)

plt.title('Monte Carlo Terminal Return Distribution (2026-10-25)')
plt.xlabel('Simulated Terminal Return (%)')
plt.ylabel('Number of Simulations')
plt.legend()
plt.tight_layout()
plt.show()

print(f'Conservative case: below {return_33:.2f}%')
print(f'Base case: {return_33:.2f}% to {return_67:.2f}%')
print(f'Optimistic case: above {return_67:.2f}%')

# simulated price distribution
# 33rd and 67th percentiles
price_33 = new_terminal_price.quantile(1 / 3)
price_67 = new_terminal_price.quantile(2 / 3)

plt.axvline(
    price_33,
    label=f'33.33rd percentile: {price_33:.2f}',
    color='red',
    linestyle='--'
)

plt.axvline(
    price_67,
    label=f'66.67th percentile: {price_67:.2f}',
    color='purple',
    linestyle='--'
)

plt.hist(
    new_terminal_price,
    bins=50,
    edgecolor='black'
)

plt.title('Monte Carlo Terminal Price (2026-10-25)')
plt.xlabel('Simulated Terminal Price ($)')
plt.ylabel('Number of Simulations')
plt.legend()
plt.tight_layout()
plt.show()

print(f'Conservative case: below ${price_33:.2f}')
print(f'Base case: ${price_33:.2f} to ${price_67:.2f}')
print(f'Optimistic case: above ${price_67:.2f}')