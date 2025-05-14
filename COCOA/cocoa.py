import requests
import pandas as pd
import os
import plotly.express as px
from datetime import datetime

from ib_insync import *

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

import time

flag_DWLD_futures = False
flag_DWLD_power_nasa = False

if flag_DWLD_futures:
    # Define file path
    file_path = "./COCOA/futures_contract_specs.csv"  # Update with your file location

    # Read the CSV file
    fut_specs = pd.read_csv(file_path).to_dict(orient='records')

    # Create a contract for E-mini S&P 500 futures

    fut_dict = [i for i in fut_specs if i['symbol'] == 'CC'][0]

    # Convert date to string if needed
    fut_dict['lastTradeDateOrContractMonth'] = str(fut_dict['lastTradeDateOrContractMonth'])

    # Create a general contract
    contract = Contract(**fut_dict)

    # Set time range
    start_date = datetime(2023, 6, 1)
    end_date = datetime.today()
    delta = timedelta(days=7)

    # Collect data in chunks
    all_bars = []

    current_end = end_date
    while current_end > start_date:
        current_start = current_end - delta
        print(f"Requesting: {current_start} to {current_end}")

        bars = ib.reqHistoricalData(
            contract,
            endDateTime=current_end.strftime('%Y%m%d %H:%M:%S'),
            durationStr='7 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1,
            keepUpToDate=False
        )

        if not bars:
            print("No data for this window. Possibly before contract was active.")
        else:
            all_bars.extend(bars)

        current_end = current_start
        time.sleep(1.5)  # Avoid pacing violations

    # Convert to DataFrame
    df = util.df(all_bars)
    df = df.drop_duplicates(subset=['date']).sort_values(by='date')

    # Save to CSV
    df.to_csv('./COCOA/CCK5_minute_data.csv', index=False)
    print(f"Saved {len(df)} minute bars to cocoa_minute_data.csv")

    # Ensure 'date' is datetime and set as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Resample to 15-minute intervals
    df_15min = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'  # If volume is present
    })

    # Drop rows with no data (e.g. during non-trading hours)
    df_15min.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

    # Reset index if needed
    df_15min = df_15min.reset_index()


    # Helper function to get historical data
    def get_data(contract):
        # Connect to IB Gateway or TWS
        ib = IB()
        ib.connect('127.0.0.1', 7496, clientId=1)  # Paper trading

        bars = ib.reqHistoricalData(
            contract=contract,  # The contract object
            endDateTime="",  # End time ("" = current time)
            durationStr="2 Y",  # Duration (e.g., "1 D" = 1 day)
            barSizeSetting="1 day",  # Granularity (e.g., "1 min", "5 mins")
            whatToShow="TRADES",  # Data type: "TRADES", "BID", etc.
            useRTH=1,  # Regular Trading Hours only
            formatDate=1,  # Date format: 1 = human-readable, 2 = UNIX
            keepUpToDate=False,  # Keep receiving live updates (False for static)
            chartOptions=[],
        )

        df = util.df(bars)

        ib.disconnect()

        return df if not df.empty else None

    df_CCN5= get_data(contract=contract)
    # Ensure 'date' is datetime in df_CCN5

    df_CCN5.to_csv("./COCOA/futures_data/CCN5.csv", index=False)

df_CCN5 = pd.read_csv("./COCOA/futures_data/CCN5.csv")
df_CCN5['date'] = pd.to_datetime(df_CCN5['date'])

# Country producers
producers = pd.read_csv("./COCOA/extended_cocoa_production_by_region.csv")
producers['Area (Mha) %'] = producers['Area (Mha)']/producers['Area (Mha)'].sum()*100

producers.sort_values(by='Area (Mha) %',
                      axis=0,
                      ascending=False,
                      inplace=True, kind='quicksort', na_position='last')

producers['(Mha) % cumsum'] =  producers['Area (Mha) %'].cumsum()
producers.reset_index(drop=True, inplace=True)


###  Weather data

# | Variable Name        | Description                  | Code                |
# | -------------------- | ---------------------------- | ------------------- |
# | T2M                  | Temperature at 2 meters (°C) | `T2M`               |
# | PRECTOT              | Precipitation total (mm/day) | `PRECTOT`           |
# | RH2M                 | Relative humidity at 2m (%)  | `RH2M`              |
# | ALLSKY_SFC_SW_DWN    | Solar irradiance (W/m²)      | `ALLSKY_SFC_SW_DWN` |

# (Mha) area in millition hestares

if flag_DWLD_power_nasa:
    # download rain data
    for region in producers.Region:

        country = producers.query('Region == @region').Country.values[0]
        log_string = "Downloading data for " + region + ", " + country + "...\n"
        print(log_string)

        lat = producers.query('Region == @region').Latitude.values[0]
        lon = producers.query('Region == @region').Longitude.values[0]

        start = "20230101"
        end = "20250430"

        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"parameters=T2M,PRECTOT,RH2M&community=AG&latitude={lat}&longitude={lon}"
            f"&start={start}&end={end}&format=JSON"
        )

        r = requests.get(url)
        data = r.json()

        # Convert the data into a DataFrame
        df = pd.DataFrame.from_dict(data['properties']['parameter'])

        # Rename the columns to 'RH2M', 'T2M', 'PRECT'
        df.columns = ['RH2M', 'T2M', 'PRECTOT']

        # Reset the index to bring dates back as a column
        df = df.reset_index()

        # Rename the index column to 'date'
        df = df.rename(columns={"index": "date"})

        # Convert the date column to datetime format
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')


        # Melt the DataFrame to long format
       # df_long = df.melt(id_vars=["date"], value_vars=["T2M", "PRECTOTCORR", "RH2M"],
       #                   var_name="variable", value_name="value")
        file_name = "./COCOA/PN_data/" + "power_nasa_" + region + "_" + country + ".csv"

        df.to_csv(file_name, index=False)

# Initialize an empty list to collect DataFrames
precipitation_dfs = []
temperature_dfs = []

for _, row in producers.iterrows():
    region = row['Region']
    country = row['Country']
    area = row['Area (Mha) %']

    # Build the filename based on the pattern
    file_path = f"./COCOA/PN_data/power_nasa_{region}_{country}.csv"

    # Check if the file exists
    if os.path.exists(file_path):
        # Read the file and parse the date column if needed
        df = pd.read_csv(file_path)

        # Add metadata to track origin
        df['Region'] = region
        df['Country'] = country
        df['area_mha_perc'] = str(area)

        # Ensure date is in datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Convert PRECTOT and area_mha_perc to numeric, coercing errors
        df['PRECTOT'] = pd.to_numeric(df['PRECTOT'], errors='coerce')
        df['T2M'] = pd.to_numeric(df['T2M'], errors='coerce')

        df['area_mha_perc'] = pd.to_numeric(df['area_mha_perc'], errors='coerce')

        # Keep only the relevant columns
        if 'PRECTOT' in df.columns:
            df_precip = df[['PRECTOT']].copy()
            df_precip['date'] = df.index if df.index.name else df.get('date', None)
            df_precip['Region'] = region
            df_precip['Country'] = country
            df_precip['area_mha_perc'] = area
            precipitation_dfs.append(df_precip)
        else:
            print(f"PRECTOT not found in file: {file_path}")

        # Keep only the relevant columns
        if 'T2M' in df.columns:
            df_t2m = df[['T2M']].copy()
            df_t2m['date'] = df.index if df.index.name else df.get('date', None)
            df_t2m['Region'] = region
            df_t2m['Country'] = country
            df_t2m['area_mha_perc'] = area
            temperature_dfs.append(df_t2m)
        else:
            print(f"T2M not found in file: {file_path}")
    else:
        print(f"File not found: {file_path}")

# Concatenate all precipitation dataframes
precipitation_all = pd.concat(precipitation_dfs, ignore_index=True)
temperature_all = pd.concat(temperature_dfs, ignore_index=True)


# Optional: Pivot if you want Region-wise columns
precip_pivot = precipitation_all.pivot(index='date', columns='Region', values='PRECTOT')
temperature_pivot = temperature_all.pivot(index='date', columns='Region', values='T2M')

region = 'Bas-Sassandra'

df = pd.concat([precip_pivot[region], temperature_pivot[region]], axis=1)
df.columns = ['PRECTOT', 'T2M']

# Merge on the 'date' column
merged_df = pd.merge(df.reset_index(), df_CCN5, on='date', how='inner')
merged_df = merged_df[merged_df.date > "2024-04-01"]

### Feature Engineering Pipeline


class CocoaFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lag_days = [1, 3, 7, 14]
        self.rolling_windows = [3, 7, 14]

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.copy()

        # --- Time Features ---
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['dayofweek'] = df['date'].dt.dayofweek
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

        # --- Lag Features ---
        for lag in self.lag_days:
            df[f'close_lag{lag}'] = df['close'].shift(lag)

        # --- Rolling Statistics ---
        for window in self.rolling_windows:
            df[f'T2M_roll_mean_{window}'] = df['T2M'].rolling(window).mean()
            df[f'PRECTOT_roll_std_{window}'] = df['PRECTOT'].rolling(window).std()
            df[f'close_roll_mean_{window}'] = df['close'].rolling(window).mean()

        # --- Differences ---
        df['delta_T2M'] = df['T2M'].diff()
        df['delta_PRECTOT'] = df['PRECTOT'].diff()

        # --- Interaction Terms ---
        df['temp_precip_interaction'] = df['T2M'] * df['PRECTOT']
        df['rolling_volatility_interaction'] = (
            df['T2M'].rolling(7).std() * df['PRECTOT'].rolling(7).std()
        )

        # --- Market Features ---
        if all(x in df.columns for x in ['open', 'high', 'low', 'volume']):
            df['intraday_range'] = df['high'] - df['low']
            df['gap'] = df['open'] - df['close'].shift(1)
            df['avg_volume_7'] = df['volume'].rolling(7).mean()

        # --- Agronomic Threshold Features ---
        df['rain_30d_sum'] = df['PRECTOT'].rolling(30).sum()
        df['rain_90d_sum'] = df['PRECTOT'].rolling(90).sum()
        df['rain_30d_days_below_5mm'] = (df['PRECTOT'] < 5).rolling(30).sum()
        df['heat_days_above_32C'] = (df['T2M'] > 32).rolling(30).sum()

        df['drought_flag'] = (df['rain_30d_sum'] < 100).astype(int)
        df['waterlogging_risk'] = (df['rain_30d_sum'] > 400).astype(int)
        df['heat_stress_flag'] = (df['heat_days_above_32C'] >= 5).astype(int)
        df['dry_spell_flag'] = (df['rain_30d_days_below_5mm'] > 10).astype(int)

        # --- Lagged Agronomic Flags ---
        df['drought_flag_lag60'] = df['drought_flag'].shift(60)
        df['heat_stress_flag_lag60'] = df['heat_stress_flag'].shift(60)
        df['rain_3mo_lag'] = df['rain_90d_sum'].shift(90)

        # --- Future Return and Direction Target ---
        df['future_return_3d'] = df['close'].shift(-3) / df['close'] - 1
        df['target_direction_3d'] = (df['future_return_3d'] > 0).astype(int)
        df['target_signal_3d'] = np.where(df['future_return_3d'] > 0.01, 1,
                                          np.where(df['future_return_3d'] < -0.01, -1, 0))

        # --- Drop early rows with NaNs from rolling/lags ---
        df = df.dropna().reset_index(drop=True)
        return df

def evaluate_directional_model(y_true, y_pred, probs=None, returns=None):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

    if returns is not None:
        strategy_returns = returns * np.sign(np.array(y_pred))
        cum_returns = np.cumsum(strategy_returns)

        plt.figure(figsize=(10, 4))
        plt.plot(cum_returns, label='Strategy Cumulative Returns')
        plt.axhline(0, color='grey', linestyle='--')
        plt.title("Backtest: Strategy Performance")
        plt.legend()
        plt.tight_layout()
        plt.show()

        sharpe = strategy_returns.mean() / strategy_returns.std()
        print(f"\nStrategy Sharpe Ratio: {sharpe:.2f}")


# Initialize and apply the feature engineer
fe = CocoaFeatureEngineer()
df_features = fe.fit_transform(merged_df)

# Define feature set and target
target_col = 'close'
exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'average', 'barCount']
feature_cols = [col for col in df_features.columns if col not in exclude_cols]

X = df_features[feature_cols]
y = df_features[target_col]

model_pipeline = Pipeline([
    ('scaler', StandardScaler()),           # Scale features
    ('regressor', Ridge(alpha=1.0))         # Ridge Regression
])

tscv = TimeSeriesSplit(n_splits=5)

rmse_scores = []
mae_scores = []
r2_scores = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model_pipeline.fit(X_train, y_train)
    preds = model_pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)

print("Cross-Validated RMSE: {:.2f} ± {:.2f}".format(np.mean(rmse_scores), np.std(rmse_scores)))
print("Cross-Validated MAE : {:.2f} ± {:.2f}".format(np.mean(mae_scores), np.std(mae_scores)))
print("Cross-Validated R²  : {:.3f} ± {:.3f}".format(np.mean(r2_scores), np.std(r2_scores)))

ridge_model = model_pipeline.named_steps['regressor']

# Coefficients
print("Coefficients:", ridge_model.coef_)

# Fit on all data
model_pipeline.fit(X, y)

# Predict next-day price using latest row
latest_X = X.iloc[[-1]]
predicted_next_close = model_pipeline.predict(latest_X)
print("Predicted next-day close:", predicted_next_close[0])


# Access the trained Ridge model and the scaler
ridge_model = model_pipeline.named_steps['regressor']
scaler = model_pipeline.named_steps['scaler']

# Get the coefficients and match to feature names
coefs = pd.Series(ridge_model.coef_, index=X.columns)

# Sort by absolute value for importance
important_features = coefs.abs().sort_values(ascending=False)

print("\nMost important regressors (by absolute value):")
print(important_features)

def backtest_model(model_pipeline, X, y, n_splits=5, min_train_size=100):
    n_samples = len(X)
    test_size = (n_samples - min_train_size) // n_splits
    results = []

    for i in range(n_splits):
        train_end = min_train_size + i * test_size
        test_end = train_end + test_size

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        model_pipeline.fit(X_train, y_train)
        preds = model_pipeline.predict(X_test)

        results.append({
            "fold": i + 1,
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds)
        })

    return pd.DataFrame(results)

results_df = backtest_model(model_pipeline, X, y, n_splits=5, min_train_size=150)
print(results_df)
print("Average RMSE:", results_df['rmse'].mean())

# Example: direction accuracy (long/short)
df_eval = X.copy()
df_eval['y_true'] = y
df_eval['y_pred'] = model_pipeline.predict(X)

# Signal: predicted return > 0
df_eval['signal'] = df_eval['y_pred'].diff().apply(lambda x: 1 if x > 0 else -1)
df_eval['actual_return'] = df_eval['y_true'].diff()

# Strategy return
df_eval['strategy_return'] = df_eval['signal'].shift(1) * df_eval['actual_return']
df_eval[['actual_return', 'strategy_return']].cumsum().plot(title='Cumulative Returns')



