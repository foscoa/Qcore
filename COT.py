from unittest.mock import inplace

import pandas as pd
import requests
import plotly.express as px
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

# File URL
# Read as if it's a CSV
cot_data = pd.read_csv("COT_data/FinFutYY_2024.txt", delimiter=",")

# query only one COT
# cot_data = cot_data.query("As_of_Date_In_Form_YYMMDD == 241217")

# query instruments
instr = ['S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE']
cot_data = cot_data.query("Market_and_Exchange_Names in @instr")

def pie_chart_plot():
    # analyse open interest
    df = cot_data.sort_values(by="Open_Interest_All", ascending=False)

    # Calculate percentages
    df['Percentage'] = (df.Open_Interest_All / df.Open_Interest_All.sum()) * 100

    # Filter out categories with less than 2%
    filtered_df = df[df['Percentage'] >= 1]

    # Create a pie chart
    fig = px.pie(filtered_df, names='Market_and_Exchange_Names', values='Open_Interest_All', title='Open Interest by Future Contract')

    # Show the chart
    fig.show()

# Convert the 'date' column to datetime format AND shift date by 3 days becasue COT is only reported on Fridays
cot_data['date'] = pd.to_datetime(cot_data.As_of_Date_In_Form_YYMMDD, format='%y%m%d') + pd.DateOffset(days=3)

# Optional: Set the converted column as the index if you want a time series
cot_data.set_index('date', inplace=True)
cot_data = cot_data.sort_index()

# Exposures COT
exposure = pd.DataFrame()
exposure['Dealer'] = cot_data.Dealer_Positions_Long_All \
                     - cot_data.Dealer_Positions_Short_All

exposure['Asset_Mgr'] = cot_data.Asset_Mgr_Positions_Long_All \
                         - cot_data.Asset_Mgr_Positions_Short_All

exposure['Lev_Money'] = cot_data.Lev_Money_Positions_Long_All \
                         - cot_data.Lev_Money_Positions_Short_All

exposure['Other_Rept_Positions'] = cot_data.Other_Rept_Positions_Long_All \
                         - cot_data.Other_Rept_Positions_Short_All

exposure_pct = (exposure/exposure.shift(1)-1)*np.sign(exposure)

# Define the ticker symbol
ticker = "SPY"

# Download data with yfinance
data = yf.download(ticker, start=exposure.index[0], end=exposure.index[-1]+ pd.DateOffset(days=1))
SPY = pd.DataFrame(data=data.reset_index()['Adj Close'].values,
                   index=data.reset_index()['Date'],
                   columns=['SPY'])

# Extract the adjusted close column
train_df = exposure.join(SPY).dropna()
train_pct = (train_df/train_df.shift(1)-1)*np.sign(train_df).dropna()
train_pct.dropna(inplace=True)

# Initialize the scaler
scaler = StandardScaler()

# Define independent variables (X) and dependent variable (y)
X = scaler.fit_transform(train_df[['Dealer', 'Asset_Mgr', 'Lev_Money', 'Other_Rept_Positions']].drop(index=train_df.index[-1]).values) # Independent variables (features)
y = train_df['SPY'].shift(-1).dropna().values  # Dependent variable (target)
y = (y - y.mean())/y.std()

# Add a constant to the model (for the intercept term)
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())