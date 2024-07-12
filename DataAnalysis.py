import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:/Users/hp/Downloads/nse_all_stock_data (1).csv/nse_all_stock_data (1).csv")    #Import dataset

# Data Cleaning
data['Date'] = pd.to_datetime(data['Date'])   # Convert 'Date' to datetime

data.fillna((data.mean()), inplace=True)  # Fill missing values with mean 
data.fillna(0) # Fill remaining missing values with 0

data.describe() # Displays basic statistics for each stock

# Basic Visualization of Data
tenstocks = data.columns[1:11]

plt.figure(figsize=(14, 7))
for stock in tenstocks:
    plt.plot(data['Date'], data[stock], label=stock)

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Trends of the First 10 Stocks')
plt.legend()
plt.show()

# Correlation Analysis
returns = data.set_index('Date').pct_change(fill_method=None)  #Calculate daily returns

correlation_matrix = returns.corr()  # create a correlation matrix
correlation_matrix.fillna(0)

c = correlation_matrix.unstack() # find pairs
correlated_pairs = c.sort_values(ascending=False).drop_duplicates().head(10)

print(correlated_pairs)

plt.figure(figsize=(14, 10)) # Plot the heatmap
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Stock Returns Correlation Heatmap')
plt.show()

# Time Series Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

stock = data.columns[1] # selecting a stock for decomposition 
stock_data = data[stock]

decomposition = seasonal_decompose(stock_data, model='additive', period=365)  # Time series decomposition

trend = decomposition.trend # Identify trend, seasonality, and residual components
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(14, 10)) # implementation

plt.subplot(411)
plt.plot(stock_data, label='Original')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Anomaly Detection
mean_returns = returns.mean()
std_returns = returns.std()

z_scores = (returns - mean_returns) / std_returns  # Calculate Z-scores of daily returns manually
z_scores.fillna(0).head()

anomalies = z_scores.unstack().sort_values(ascending=False).head(5)  # Find the significant anomalies using Z-scores
print(anomalies)

anomalies_details = []  # details of anomalies
for idx, z_score in anomalies.items():
    stock, date = idx
    price_change = returns.loc[date, stock]
    anomalies_details.append((stock, date, price_change, z_score))

for detail in anomalies_details:
    print(f"\nStock: {detail[0]}, Date: {detail[1]}, Price Change: {detail[2]:.4f}, Z-score: {detail[3]:.4f}")
