#Multifactor Regression

#Using data to predict OXM stock trends
#XRT(retail sector stock) GSPC(S&P 500~overall market) TNX(10-year Treasurt yield, macro/rates)

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

#OXM,XRT,^GSPC,^TNX Stock History
tickers = ["OXM","XRT","^GSPC","^TNX"]
all_values = []
for ticker in tickers:
    value = yf.Ticker(ticker)
    all_values.append(value)
all_histories = []
for ticker in all_values:
    history = ticker.history(period="1y")
    all_histories.append(history)
all_closing_prices = []
for history in all_histories:
    closing_prices = history["Close"]
    closing_prices.index = closing_prices.index.tz_localize(None)
    all_closing_prices.append(closing_prices)
print(all_closing_prices)

#Merge DataFrames
combined = pd.concat(all_closing_prices, axis=1, keys=tickers)
combined = combined.dropna()
correlation_matrix = combined.corr()
print(correlation_matrix)

#            OXM       XRT     ^GSPC      ^TNX
#OXM    1.000000  0.192054  0.184529  0.213338
#XRT    0.192054  1.000000  0.399307  0.177346
#^GSPC  0.184529  0.399307  1.000000  0.746922 # NOTE
#^TNX   0.213338  0.177346  0.746922  1.000000

# X = predictors, y = target
X = combined[["XRT", "^GSPC", "^TNX"]]
y = combined["OXM"]

# Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Check performance
r_squared = model.score(X_test, y_test)
print(f"R-squared: {r_squared:.3f}")

# Feature importance = the coefficients
coefficients = pd.DataFrame({
    "Predictor": X.columns,
    "Coefficient": model.coef_
})
print(coefficients)