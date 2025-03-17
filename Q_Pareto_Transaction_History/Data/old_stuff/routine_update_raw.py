import pandas as pd
import numpy as np
from pathlib import Path
import os
import datetime

# Define the file path raw data
file_path_raw = "Q_Pareto_Transaction_History/Data/TradeHistory_raw.csv"
file_path_archive = "Q_Pareto_Transaction_History/Data/Archive"

# Get list of files
folder_path = Path(file_path_archive)

# Get the most recently modified file
latest_file = max(folder_path.glob("*.csv"), key=lambda f: f.stat().st_mtime, default=None)

## CHECK LATEST MODIFIED FILE
# Get modification time
mod_time = os.path.getmtime(latest_file)
mod_date = datetime.datetime.fromtimestamp(mod_time).date()

# Get today's date
today = datetime.date.today()

assert mod_date == today, 'The most recently modified file was not updated today.'
print("The last file appended is: " + str(latest_file))

## READ NEW FILES
# Read the CSV file raw
df_raw = pd.read_csv(file_path_raw)
df_new = pd.read_csv(latest_file)

# check for overlapping periods
first_date_new = pd.to_datetime(df_new.DateTime.dropna().str.replace(";", " "), format="%Y%m%d %H%M%S").min()
max_date_raw = pd.to_datetime(df_raw.DateTime.dropna().str.replace(";", " "), format="%Y%m%d %H%M%S").max()
assert first_date_new > max_date_raw, 'The first date in the new data is older than the last date in the existing data'

#check if columns are the same
assert bool((df_raw.columns == df_new.columns).sum()), 'The new and old dfs have different columns'

df_to_write = pd.concat([df_new, df_raw], axis=0, ignore_index=True)
df_to_write.to_csv("Q_Pareto_Transaction_History/Data/TradeHistory_raw.csv", index=False)  # index=False prevents writing row indices)



