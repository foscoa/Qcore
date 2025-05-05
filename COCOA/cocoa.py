import requests
import pandas as pd

import plotly.express as px
from datetime import datetime

from ib_insync import *

# Connect to IB Gateway or TWS
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)  # Paper trading

# Define file path
file_path = "../futures_contract_specs.csv"  # Update with your file location

# Read the CSV file
fut_specs = pd.read_csv(file_path).to_dict(orient='records')

# Create a contract for E-mini S&P 500 futures

fut_dict = [i for i in fut_specs if i['symbol'] == 'CC'][0]

# Convert date to string if needed
fut_dict['lastTradeDateOrContractMonth'] = str(fut_dict['lastTradeDateOrContractMonth'])

# Create a general contract
contract = Contract(**fut_dict)



# Helper function to get historical data
def get_data(contract):
    bars = ib.reqHistoricalData(
        contract=contract,  # The contract object
        endDateTime="",  # End time ("" = current time)
        durationStr="1 Y",  # Duration (e.g., "1 D" = 1 day)
        barSizeSetting="1 day",  # Granularity (e.g., "1 min", "5 mins")
        whatToShow="TRADES",  # Data type: "TRADES", "BID", etc.
        useRTH=1,  # Regular Trading Hours only
        formatDate=1,  # Date format: 1 = human-readable, 2 = UNIX
        keepUpToDate=False,  # Keep receiving live updates (False for static)
        chartOptions=[]
    )

    df = util.df(bars)
    return df if not df.empty else None

df = get_data(contract=contract)

# Country producers
producers = pd.read_csv("./COCOA/extended_cocoa_production_by_region.csv")
producers['Area (Mha) %'] = producers['Area (Mha)']/producers['Area (Mha)'].sum()*100

producers.sort_values(by='Area (Mha) %',
                      axis=0,
                      ascending=False,
                      inplace=True, kind='quicksort', na_position='last')

producers['(Mha) % cumsum'] =  producers['Area (Mha) %'].cumsum()
df.reset_index(drop=True, inplace=True)


###  Weather data

# | Variable Name        | Description                  | Code                |
# | -------------------- | ---------------------------- | ------------------- |
# | T2M                  | Temperature at 2 meters (°C) | `T2M`               |
# | PRECTOT              | Precipitation total (mm/day) | `PRECTOT`           |
# | RH2M                 | Relative humidity at 2m (%)  | `RH2M`              |
# | ALLSKY_SFC_SW_DWN    | Solar irradiance (W/m²)      | `ALLSKY_SFC_SW_DWN` |

# create an aggreagate measure



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

    # Rename the columns to 'RH2M', 'T2M', 'PRECTOTCORR'
    df.columns = ['RH2M', 'T2M', 'PRECTOTCORR']

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

df_curr = pd.read_csv("./COCOA/PN_data/power_nasa_Esmeraldas_Ecuador.csv",
                 index_col='date')

# Plot the data using Plotly
fig = px.line(df, x='date', y='value', color='variable',
              labels={'date': 'Date', 'value': 'Value', 'variable': 'Variables'},
              title='Weather Data (Temperature, Precipitation, Humidity)')

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Value',
    legend_title='Variables'
)

fig.show()