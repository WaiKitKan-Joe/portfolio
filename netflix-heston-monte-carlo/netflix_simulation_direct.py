import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import requests

particular_date = pd.Timestamp.today().strftime('%Y-%m-%d')
url = (f"https://api.nasdaq.com/api/quote/NFLX/historical?"
       f"assetclass=stocks&fromdate=2021-10-21&limit=9999"
       f"&todate={particular_date}&random=28")

headers = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json, text/plain, */*',
    'Origin': 'https://www.nasdaq.com',
    'referer': 'https://www.nasdaq.com/'
}

response = requests.get(url, headers=headers)
data = response.json()

print(data.keys())
print(data['data'].keys())
print(data['data']['symbol'])
print(data['data']['totalRecords'])
print(data['data']['tradesTable'].keys())
print(data['data']['tradesTable']['asOf'])
print(data['data']['tradesTable']['headers'])
print(data['data']['tradesTable']['rows'])

df = pd.DataFrame(
    data['data']['tradesTable']['rows']
)

df = df.rename(columns={'close/Last':'close'})

column_to_be_dropped = [
    'open',
    'high',
    'low',
    'volume'
]

df = df.drop(columns=column_to_be_dropped)

df['close'] = pd.to_numeric(
    df['close']
    .astype(str)
    .replace('[$]','',regex=True),
    errors='coerce'
)

df['date'] = pd.to_datetime(
    df['date'],
    format='%m/%d/%Y',
)

df = df.set_index('date')
df = df.sort_index()


# calculate log return
df['Log close price'] = np.log(df['close'])
df['Log return'] = (df['Log close price'] -
                    df['Log close price'].shift(1))

df['Rolling volatility'] = (
    df['Log return'].rolling(window=20).std() * np.sqrt(252)
)
df['Rolling variance'] = df['Rolling volatility'] ** 2


# plot stock price movement
plt.plot(
    df.index,
    df['Log return']
)
plt.title(
    f"Netflix Daily Log Returns: "
    f"{df.index.min().date()} to "
    f"{df.index.max().date()}"
)

plt.xlabel('date')
plt.ylabel('Log return')
plt.tight_layout()
plt.show()

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


if not np.isfinite(rho):
    rho = 0.0

rho = np.clip(rho, -0.999, 0.999)

mu = (
    df['Log return'].mean() * 252 + 0.5 * np.mean(
        df['Previous variance'])
)

v0 = df['Rolling variance'].iloc[-1]

# simulate
start_date = pd.Timestamp(df.index[-1])
end_date = start_date + pd.offsets.BDay(10)

date = pd.bdate_range(
    start=start_date,
    end=end_date
)

number_of_step = len(date) - 1
length_of_date = len(date)


number_of_path = 50000
initial_price = df['close'].iloc[-1]

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

    new_variance = (
            previous_variance
            + kappa * (theta - previous_variance) * dt
            + xi
            * np.sqrt(previous_variance * dt)
            * variance_z
    )

    variance_path[t] = np.maximum(
        new_variance,
        0
    )

    stock_path[t] = (
        previous_price
        * np.exp(
        (mu - 0.5 * previous_variance) * dt
        + np.sqrt(previous_variance * dt) * stock_z)
    )

stock_df = pd.DataFrame(
    stock_path,
    index=date
)
variance_df = pd.DataFrame(
    variance_path,
    index=date
)
# investment analysis
new = stock_df.copy()

new_terminal_price = new.iloc[-1]
new_initial_price = new.iloc[0]

# plot MCS
plt.plot(
    stock_df.loc[
        stock_df.index.to_series().between(
            start_date, end_date)]
)
plt.title('Netflix Heston Monte Carlo Price Paths')
plt.xlabel('Trading day')
plt.ylabel('Simulated stock price')
plt.tight_layout()
plt.show()

# investment analysis
new_return = new_terminal_price / new_initial_price - 1
mean_new_terminal_price = new_terminal_price.mean()

new_return_avg = new_return.mean()
simulated_returns = new_return

new_price_median = new_terminal_price.median()
new_return_median = new_return.median()

prob_profit = np.mean(new_terminal_price >new_initial_price)

print(f'\nInvestment analysis for {start_date} and {end_date}:')
print(f"Mean terminal return: {new_return_avg:.3%}")
print(f"Median terminal return: {new_return_median:.3%}")

print(f'Median terminal price: ${new_price_median:.3f}')
print(f'Median termina return between:{new_return_median:.3%}')

print(f'Probability (terminal price > initial price)'
      f':{prob_profit:.3%}')

# 95VaR
var_95_return = new_return.quantile(0.05)
var_95_loss = max(-var_95_return,0)
worst_5_pt = new_return[simulated_returns < var_95_return]
shortfall_return = max(-worst_5_pt.mean(),0)

print(f'\n95% VaR: {var_95_loss:.3%} loss')
print(f'Expected shortfall: {shortfall_return:.3%}')

# 33rd and 67th percentiles
return_33 = simulated_returns.quantile(1 / 3)
return_67 = simulated_returns.quantile(2 / 3)

plt.hist(
    simulated_returns,
    bins=50,
    edgecolor='black'
)

plt.axvline(
    return_33,
    label=f'33.33rd percentile: {return_33:.2%}',
    color='red',
    linestyle='--'
)

plt.axvline(
    return_67,
    label=f'66.67th percentile: {return_67:.2%}',
    color='purple',
    linestyle='--'
)

plt.title(f'Monte Carlo Terminal Return Distribution '
          f'{end_date.date()}')
plt.xlabel('Simulated Terminal Return (%)')
plt.ylabel('Number of Simulations')
plt.legend()
plt.tight_layout()
plt.show()

print(
    f"\nLower-third scenario: below "
    f"{return_33:.2%}"
)

print(
    f"Middle-third scenario: "
    f"{return_33:.2%} to {return_67:.2%}"
)

print(
    f"Upper-third scenario: above "
    f"{return_67:.2%}"
)

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

plt.title(f'\nMonte Carlo Terminal Price ({end_date.date()})')
plt.xlabel('Simulated Terminal Price ($)')
plt.ylabel('Number of Simulations')
plt.legend()
plt.tight_layout()
plt.show()

print(f'\nLower-third scenario: below ${price_33:.2f}')
print(f'Middle-third scenario: ${price_33:.2f} to ${price_67:.2f}')
print(f'Upper-third scenario: above ${price_67:.2f}')
