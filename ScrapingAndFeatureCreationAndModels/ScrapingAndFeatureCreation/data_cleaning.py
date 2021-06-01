import numpy as np
import pandas as pd


def drop_duplicate_rows(df):
    """
    Drops the duplicate rows in the dataframe based on Date column.

    Parameters
    ----------

    df : dataframe

    Returns
    -------

    df: dataframe
        updated dataframe after droping duplicates.

    """
    df = df.drop_duplicates(subset=["Date"], keep="first")
    return df


def fill_with_previous_values(df):
    """
    Fills the null values in the dataframe with the values from the previous row.

    Parameters
    ----------

    df : dataframe

    Returns
    -------

    df : dataframe
        updated dataframe after filling with previous values.

    """
    df.fillna(method="ffill", inplace=True)
    return df


def add_missing_rows(df, ind):
    """

    Adds rows to the stock dataframe.

    If the date is present in index dataframe an not present in stock dataframe,
    then a new row (as date and NAN values) is added to stock dataframe.

    Parameters
    ----------

    df : dataframe
        stock dataframe

    ind : dataframe
        index dataframe

    Returns
    -------

    df : dataframe
        updated dataframe after adding new rows.

    """

    df.Date = pd.to_datetime(df.Date)
    ind.Date = pd.to_datetime(ind.Date)
    s = df.Date.head(1).values[0]
    e = df.Date.tail(1).values[0]
    ind = ind[ind.Date.between(e, s)]
    missing_df = pd.DataFrame(columns=df.columns)
    indexes_dates = ind.Date.values
    df.Date = pd.to_datetime(df.Date)
    df_dates = df.Date.values
    start = 0
    for i, v in enumerate(indexes_dates):
        if v not in df.Date.values:
            m = abs(ind.shape[1]-missing_df.shape[1])
            res = list(np.append(ind.iloc[i].values, [np.nan]*m))
            missing_df.loc[start] = res
            start += 1
    df = pd.concat([df, missing_df])
    return df


def cleaning(df, ind):
    """
    Removes duplicate rows, Adds missing rows, fills null values from pervious row to the stock dataframe.

    Parameters
    ----------

    df : dataframe
        stock dataframe

    ind : dataframe
        index dataframe

    Returns
    -------

    df : dataframe
        updated dataframe after performing all the operations.

    """

    df = drop_duplicate_rows(df)
    ind = drop_duplicate_rows(ind)
    df = add_missing_rows(df, ind)
    df = fill_with_previous_values(df)
    df.reset_index(drop=True, inplace=True)
    df = drop_duplicate_rows(df)
    df = df.sort_values(by=["Date"], ascending=[False])
    return df, ind
