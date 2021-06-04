import sys
import os
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from bs4 import BeautifulSoup
import traceback
import subprocess


def download_revenue_profit(code, name):
    """
    Creates the revenue profit file.
    Parameters
    ----------
    code : string
        security code of the company.
    name : 
        security id of the company.
    Methods:
    --------
    create_driver : creates the chrome driver.
    download : extracts the data from the page and saves to a csv file.
    """


    def create_driver():
        """
        Creates a Chrome Driver.
        Returns
        --------
        driver : driver
            chrome web driver.
        """

        chromeOptions = webdriver.ChromeOptions()
        chromeOptions.add_argument("--headless")
        chromeOptions.add_experimental_option(
            "prefs", {"download.default_directory": path})
        driver = webdriver.Chrome(
            ChromeDriverManager().install(), options=chromeOptions)
        return driver

    def download():
        """
        extracts the data from the page and saves to a csv file.
        """
        columns = ["security code", "security name", 'revenue',
                   'income', 'expenditure', 'profit', 'eps', "year", "quartile"]
        code_df = pd.DataFrame(columns=columns)
        ref = pd.read_csv(path,str(code)+".csv")
        last = df.iloc[-1]
        year = last['year'] - 2007
        quater = last['quartile']
        start = (year * 4) + 52 + quater + 1
        end = datetime.datetime.now().year - 2007
        end = (end * 4) + 52 + 4
        for q in range(start, end):
            try:
                url = "https://www.bseindia.com/corporates/results.aspx?Code=" + \
                    str(code) + "&Company=" + str(name) + \
                    "&qtr=" + str(q) + "&RType=D"
                driver.get(url)
                html = driver.page_source
                soup = BeautifulSoup(html, "lxml")

                table = soup.find_all(
                    "table", attrs={"id": "ContentPlaceHolder1_tbl_typeID"})
                table = pd.read_html(str(table))[0]
                table = table[[0, 1]]
                table.dropna(inplace=True)
                table = table.transpose()
                table.columns = table.iloc[0]
                table = table[1:]
                table.columns = map(str.lower, table.columns)
                table.drop(["description"], inplace=True, axis=1)
                try:
                    table["date begin"] = pd.to_datetime(table["date begin"])
                    date = table.iloc[0]["date begin"]
                    table["quartile"] = (date.month-1)//3 + 1
                    table["year"] = date.year
                    table["security name"] = name
                    table["security code"] = code
                    cols = table.columns
                    mycols = ['revenue', 'income',
                              'expenditure', 'profit', 'eps']
                    row = {}
                    row["security name"] = name
                    row["security code"] = code
                    row["year"] = date.year
                    row["quartile"] = (date.month-1)//3 + 1
                    for my in mycols:
                        try:
                            res = [c for c in cols if my in c]
                            if my == "income":
                                p = [c for c in res if "total income" == c]
                                res = p or res
                            elif my == "profit":
                                p = [c for c in res if "net profit" == c]
                                res = p or res
                            elif my == "expenditure":
                                p = [c for c in cols if "expenses" in c]
                                res = p or res
                            elif my == "eps":
                                a = "Basic for discontinued & continuing operation"
                                b = "Diluted for discontinued & continuing operation"
                                p = [c for c in cols if a.lower()
                                     in c or b.lower() in c]
                                res = p or res
                            elif my == "revenue":
                                p = [c for c in cols if "sales" in c]
                                res = p or res
                                # row["revenue"] = table[res].values[0][0]
                                # continue
                                pass
                            row[my] = table[res].values[0][0]
                        except:
                            row[my] = ""
                            traceback.print_exc()
                    code_df = code_df.append(row, ignore_index=True)
                except:
                    traceback.print_exc()
            except:
                traceback.print_exc()
        code_df = ref.append(code_df)
        code_df = code_df.sort_values(by=["year","quartile"],ascending=[True,True])
        code_df.to_csv(os.path.join(path, str(code)+".csv"), index=None)

    driver = create_driver()
    download()

path = os.path.join(os.getcwd(),"Data","Revenue")
if not os.path.exists(path):
    os.makedirs(path)

df = pd.read_csv(os.path.join(os.getcwd(), "Data", "Equity.csv"))

for index, row in df.iterrows():
    try:
        code = row["Security Code"]
        name = row["Security Id"]
        download_revenue_profit(code, name)
    except:
        traceback.print_exc()
    break
