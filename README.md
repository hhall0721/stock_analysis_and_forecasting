Stock Analysis and Forecasting:

A collection of stock market analysis and forecasting projects, built while learning to combine financial data, SQL, and statistical modeling. Some of this started as general practice, some grew out of real prep work (see BrandData), but all of it follows the same core loop: pull price data, ask a specific question about it, and check whether the model actually answers that question or just looks like it does.

AirlineStock
Four scripts analyzing Alaska, Delta, United, and Southwest stock prices: a logistic regression classifier for next-day price direction, a SQL pipeline storing and joining price and company data, a correlation and forecasting comparison (linear, polynomial, ARIMA), and a Monte Carlo simulation using geometric Brownian motion to estimate a 90% confidence range for Alaska Airlines' price 30 days out.
Full writeup

BrandData
Stock and business analysis for Constellation Brands (STZ) and Oxford Industries (OXM, parent of Tommy Bahama), including a regional demand consolidation model built for meeting prep with a wine industry contact, and a multifactor regression testing whether market-wide and sector factors explain OXM's stock price (they mostly don't, R-squared of 0.037, which points to company-specific fundamentals instead).
Full writeup

Tools used across both: Python, yfinance, pandas, SQLite, scikit-learn, statsmodels, matplotlib, seaborn, numpy

