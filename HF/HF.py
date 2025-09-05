import pandas as pd
import numpy as np

# Config
drop_cols = [
    'Total', 'CurrencyForwards', 'HedgeFunds', 'MultistrategyHedgeFunds',
    'Conviction/Disc.Funds', 'CorrelatedHedgeFunds', 'NonHedgeFunds',
    'PrincipalTrading', 'OtherPrincipalTrading', 'CashaccountsPrincipalTrading',
    'HedgingBook', 'Safety', 'Gold', 'Cash', 'Loans', 'Participations',
    '1LCapitalAGCallOption', 'ArvyConvertible', 'QCOREFundManagementAG',
    'WineFinancing', 'QCFMCapitalisation'
]

# Reusable functions
def read_excel_file(path, header=1, index_col=1):
    df = pd.read_excel(path, header=header, index_col=index_col)
    return df.drop(df.columns[0], axis=1)

def clean_df(df, col_pattern, drop_list, fillna_value=0):
    if col_pattern != 'Currency':
        out = df.loc[:, df.columns.str.contains(col_pattern)].transpose()
        out.index = pd.to_datetime(out.index.str.split("\n").str[-1])
        out.columns = out.columns.str.split(",").str[-1].str.replace(" ", "", regex=True)
        out = out.drop(columns=[c for c in drop_list if c in out.columns], errors="ignore")
    else:
        out = df.loc[:, df.columns.str.contains(col_pattern)].transpose()
        out.columns = out.columns.str.split(",").str[-1].str.replace(" ", "", regex=True)
        out = out.drop(columns=[c for c in drop_list if c in out.columns], errors="ignore")

    return out.fillna(fillna_value)

def merge_dataframes(df_list):
    return pd.concat(df_list, axis=0, sort=False).groupby(level=0).sum(min_count=1)

def align_monthly(df1, df2, value_col):
    """
    Align df2 data to df1's dates based on year and month.

    Parameters:
    - df1: DataFrame with the desired index (dates)
    - df2: DataFrame containing the values to map
    - value_col: column name in df2 to align

    Returns:
    - DataFrame with df1's index and the aligned values
    """
    # Create a Series mapping (year, month) -> value from df2
    df2_map = df2[value_col].copy()
    df2_map.index = list(zip(df2.index.year, df2.index.month))

    # Map df1's dates using the same (year, month) key
    aligned_values = df1.index.to_series().apply(lambda x: df2_map.get((x.year, x.month), pd.NA))

    # Return a new DataFrame with df1's index
    return pd.DataFrame({value_col: aligned_values}, index=df1.index)



# Read and clean data
df1 = read_excel_file("HF/HF_MVRet_31Aug2023_30Jun2025.xlsx")
df2 = read_excel_file("HF/HF_MVRet_31Dec2021_31Jul2023.xlsx")
hc = pd.read_csv("HF/hedging_costs.csv", index_col=0)
hc.index = pd.to_datetime(hc.index)

mv_merged = merge_dataframes([clean_df(df1, "Value", drop_cols),
                              clean_df(df2, "Value", drop_cols)]).replace(0, np.nan)

ret_merged = merge_dataframes([clean_df(df1, "FIRD Loc Return", drop_cols),
                               clean_df(df2, "FIRD Loc Return", drop_cols)]).fillna(0)

ccy = pd.concat([clean_df(df1, "Currency", drop_cols),
                        clean_df(df2, "Currency", drop_cols)]).bfill().iloc[0]


# control to check if dfs have same columns
if (mv_merged.columns != ret_merged.columns).sum() > 0:
    print("columns are not the same!")

# apply hedging costs
hc_aligned = align_monthly(df1=ret_merged, df2=hc, value_col = "Cost of hedging")

ret_EUR = (ret_merged - ((ccy == 'USD'))*((0*mv_merged+1).fillna(0))*hc_aligned.values).fillna(0)

pnL = (ret_EUR/100*(mv_merged.fillna(0).shift(1))).dropna()
fund_AUM = mv_merged.fillna(0).shift(1).dropna().sum(axis=1)
fund_ret = pnL.sum(axis=1)/fund_AUM
fund_ret.to_csv("HF/fund_ret.csv")



