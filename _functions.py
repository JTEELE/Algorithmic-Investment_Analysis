# Import dependencies, read and format data for analysis:
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
from pathlib import Path
from _functions import *

def plot_beta(total_dataset):
    rolling_algo_covariance = total_dataset['Total_algo'].rolling(window=60).cov(total_dataset['SP500_Daily_return'])
    rolling_sp_variance = total_dataset['SP500_Daily_return'].rolling(window=60).var()
    rolling_algo_beta = rolling_algo_covariance/rolling_sp_variance
    rolling_algo_beta.plot(figsize=(20,10), title="Algo Beta Trend")
    return rolling_sp_variance

def sharp_ratio(total_dataset):
    # leveraging the exponential moving average:
    volatility = total_dataset.std() * np.sqrt(252)
    ewm_total_dataset = total_dataset.ewm(halflife=21).mean()
    ewm_total_dataset.plot(figsize=(20,10))
    # Annualized Sharpe Ratios
    print('Sharpe Ratios:')
    sharpe_ratios = (total_dataset.mean() * 252) / volatility
    # Visualizing the sharpe ratios as a bar plot
    sharpe_ratios.plot.bar(title='Sharpe Ratios')

def corr_heatmap(total_dataset):
    correlation = total_dataset.corr()
    sns.heatmap(correlation, vmin=-1, vmax=1)

def rolling_std(total_dataset,custom_portfolio):
    # Set weights of portfolio
    weights = [1/3, 1/3, 1/3]
    # Calculate portfolio return
    custom_portfolio_returns = custom_portfolio.dot(weights)
    cp_cumulative_returns = (1 + custom_portfolio_returns).cumprod() -1
    # Join your returns DataFrame to the original returns DataFrame
    final_portfolios = pd.concat([total_dataset,custom_portfolio,], axis='columns', join='inner')
    final_portfolios.sort_index(inplace=True)
    # Only compare dates where return data exists for all the stocks (drop NaNs)
    final_portfolios.dropna()
    final_portfolios.isnull().sum()
    # Calculate the annualized `std`
    final_volatility = final_portfolios.std() * np.sqrt(252)
    # Calculate rolling standard deviation
    final_portfolios.rolling(window=21).std()
    # Plot rolling standard deviation
    return final_portfolios