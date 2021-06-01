import pandas as pd
import re
import traceback


def bonus_issue(stock, start_date, end_date, r1, r2):
    """
    For an r1:r2 bonus shares,
    if y is the stock value before the bonus share issue,
    then the value of the stock will be y*(r2/(r1+r2)),
    for the data between the given dates.

    Parameters
    ----------

    stock : dataframe

    start_date : datetime

    end_date : datetime

    r1 : integer

    r2 : integer

    Returns
    -------

    stock : dataframe
        updated dataframe after bonus
    """
    specific_dates = stock[stock.Date.between(end_date, start_date)]
    for index, row in specific_dates.iterrows():
        specific_dates.loc[index, "Open Price"] = specific_dates.loc[index,
                                                                     "Open Price"] * (r2/(r1+r2))
        specific_dates.loc[index, "Low Price"] = specific_dates.loc[index,
                                                                    "Low Price"] * (r2/(r1+r2))
        specific_dates.loc[index, "High Price"] = specific_dates.loc[index,
                                                                     "High Price"] * (r2/(r1+r2))
        specific_dates.loc[index, "Close Price"] = specific_dates.loc[index,
                                                                      "Close Price"] * (r2/(r1+r2))
        specific_dates.loc[index,
                           "WAP"] = specific_dates.loc[index, "WAP"] * (r2/(r1+r2))
        stock.loc[index] = specific_dates.loc[index]
    return stock


def stock_split(stock, start_date, end_date, r1, r2):
    """
    For an r1:r2 stock split, if y is the stock value before the split,
    then the value of the stock will be y*(r1/r2),
    for the data between the given dates.

    Parameters
    ----------

    stock : dataframe

    start_date : datetime

    end_date : datetime

    r1 : integer

    r2 : integer

    Returns
    -------

    stock : dataframe
        updated dataframe after splitting
    """
    specific_dates = stock[stock.Date.between(end_date, start_date)]
    for index, row in specific_dates.iterrows():
        specific_dates.loc[index,
                           "Open Price"] = specific_dates.loc[index, "Open Price"] * (r1/r2)
        specific_dates.loc[index,
                           "Low Price"] = specific_dates.loc[index, "Low Price"] * (r1/r2)
        specific_dates.loc[index,
                           "High Price"] = specific_dates.loc[index, "High Price"] * (r1/r2)
        specific_dates.loc[index, "Close Price"] = specific_dates.loc[index,
                                                                      "Close Price"] * (r1/r2)
        specific_dates.loc[index,
                           "WAP"] = specific_dates.loc[index, "WAP"] * (r1/r2)
        try:
            stock.loc[index] = specific_dates.loc[index]
        except:
            traceback.print_exc()

    return stock


def create_dividend(stock, corporate):
    """
    Creates new Dividend Value column in the stock dataframe.

    Parameters
    ----------

    corporate : dataframe

    stock : dataframe

    Returns
    -------

    stock : dataframe
        updated dataframe with dividend column

    """
    corporate['Ex Date'] = pd.to_datetime(
        corporate['Ex Date'], errors='coerce')
    stock['Date'] = pd.to_datetime(stock['Date'], errors='coerce')

    dividend = corporate[corporate['Purpose'].str.contains("Dividend")]
    result = {}
    for index, row in dividend.iterrows():
        try:
            year = row["Ex Date"].year
            month = row["Ex Date"].month
            amount = re.findall(r"\d+.?\d*", row["Purpose"])[0]
            res = result.get(year, {})
            q = "1q" if 1 <= month <= 3 else "2q" if 4 <= month <= 6 else "3q" if 6 <= month <= 9 else "4q"
            val = res.get(q, [])
            val.append(float(amount))
            res[q] = val
            result[year] = res
        except:
            pass

    for year, quaters in result.items():
        for q, a in quaters.items():
            quaters[q] = sum(a)/len(a)
        result[year] = quaters
    divList = list()
    for index, row in stock.iterrows():
        year = row["Date"].year
        month = row["Date"].month
        q = "1q" if 1 <= month <= 3 else "2q" if 4 <= month <= 6 else "3q" if 6 <= month <= 9 else "4q"
        if result.get(year) != None:
            if result.get(year).get(q) != None:
                divList.append(result.get(year).get(q))
            else:
                divList.append(0)
        else:
            divList.append(0)
    stock["Dividend Value"] = divList
    return stock


def apply_corporate_actions(stock, corporate):
    """
    Applies stock split and bonus on the given stock dataset.

    creates bonus dataframe and invoke bonus_issue method.

    creates split dataframe and invoke stock_split method.

    creates dividend value Column in Stock dataframe by invoking create_dividend method.

    Parameters
    ----------

    stock : dataframe

    corporate : dataframe

    Returns
    -------

    stock : dataframe
        updated dataframe after stock split and bonus and dividend.

    Methods
    -------
    stock_split : 

    bonus_issue :

    """
    stock["Date"] = pd.to_datetime(stock["Date"])
    corporate["Ex Date"] = pd.to_datetime(
        corporate["Ex Date"], errors='coerce')
    # corporate["BC Start Date"] = pd.to_datetime(corporate["BC Start Date"],errors='coerce')
    # corporate[" BC End Date\t"] = pd.to_datetime(corporate[" BC End Date\t"],errors='coerce')
    # corporate["ND Start Date"] = pd.to_datetime(corporate["ND Start Date"],errors='coerce')
    # corporate["ND End Date"] = pd.to_datetime(corporate["ND End Date"],errors='coerce')

    bonus_df = corporate[corporate['Purpose'].str.contains("Bonus")]
    for index, row in bonus_df.iterrows():
        start_date = bonus_df.loc[index, "Ex Date"]
        ratio = bonus_df.loc[index, "Purpose"]
        r1, r2 = re.findall(r"\d+", ratio)
        r1, r2 = int(r1), int(r2)
        end_date = stock.tail(1)["Date"].values[0]
        stock = bonus_issue(stock, start_date, end_date, r1, r2)

    stock_split_df = corporate[corporate['Purpose'].str.contains("Stock")]
    for index, row in stock_split_df.iterrows():
        start_date = stock_split_df.loc[index, "Ex Date"]
        ratio = stock_split_df.loc[index, "Purpose"]
        r1, r2 = re.findall(r"\d+", ratio)
        r1, r2 = int(r1), int(r2)
        end_date = stock.tail(1)["Date"].values[0]
        stock = stock_split(stock, start_date, end_date, r1, r2)

    stock = create_dividend(stock, corporate)

    return stock
