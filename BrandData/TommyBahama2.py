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

# NOTE: Had to strip timezone info before merging — ^TNX pulls from a
# different data source (bond market) than equities, so it was tagged
# with a different UTC offset than OXM/XRT/GSPC even on the "same" date.
# Used .tz_localize(None) to fix. Without this, concat silently returned
# 0 overlapping rows instead of an error.

# CORRELATION MATRIX FINDINGS:
# - OXM correlates weakly with all 3 predictors (0.18-0.21)
# - ^GSPC and ^TNX are correlated at 0.75 (multicollinearity risk)
#   Interpretation: rates and market normally move opposite (higher rates
#   should lower stock valuations), but here both rose together —
#   suggests investors are pricing in economic growth, not rate fear.

# MODEL RESULTS:
# R-squared = 0.037 (model explains ~3.7% of OXM price movement)
# Honest takeaway: broad market/sector/rate conditions don't meaningfully
# predict OXM specifically. Consistent with weak correlations above.
# Likely story: company-specific fundamentals (see F&B segment revenue
# analysis) drive OXM more than macro conditions do.

# TODO NEXT SESSION:
# 1. Standardize predictors (mean 0, std 1) before comparing coefficients —
#    raw coefficients aren't comparable across variables on different scales
#    (e.g. GSPC coefficient looks tiny only because GSPC values are in the
#    thousands while TNX values are in single digits)
# 2. Check VIF (variance inflation factor) to quantify how much the
#    GSPC/TNX correlation (0.75) is distorting the model