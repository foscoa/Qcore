import pandas as pd
import requests
import plotly.express as px

# File URL
# Read as if it's a CSV
cot_data = pd.read_csv("COT_data/FinFutYY_2024.txt", delimiter=",")

# query only one COT
cot_data = cot_data.query("As_of_Date_In_Form_YYMMDD == 241217")

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

cot_data['Commercial_Exposure'] = cot_data.Dealer_Positions_Long_All \
                                - cot_data.Dealer_Positions_Short_All \
                                + cot_data.Asset_Mgr_Positions_Long_All \
                                - cot_data.Asset_Mgr_Positions_Short_All