import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import datetime
import calendar
import zipfile
import json


def download_deliverables():
    """
    Downloads the deliverables of the current day.

    deliverables_url = "https://www.bseindia.com/markets/equity/EQReports/GrossShortPos.aspx?flag=0"

    creates the driver.

    opens the deliverables_url.

    downloads the file.

    Methods:
    --------

    create_driver : creates the chrome driver.

    download : downloads the file. invokes extract_save method.

    extract_save : extracts the file and renames it to the specified name.

    """

    deliverables_url = "https://www.bseindia.com/markets/equity/EQReports/GrossShortPos.aspx?flag=0"

    path = os.path.join(os.getcwd(), "Data", "Stock")

    def extract_save(name):
        """
        extracts the file and renames it to the specified name.

        Parameters
        ----------
        name : string
            name of the file i.e, security id

        """

        res = zipfile.ZipFile(os.path.join(path, name))
        res.extractall(path)
        deli = pd.read_csv(os.path.join(path, name), sep="|",
                           converters={'DATE': lambda x: str(x)})
        deli["DATE"] = deli["DATE"].apply(
            lambda row: datetime.datetime.strptime(str(row), "%d%m%Y").date())
        deli.to_csv(os.path.join(path, "deliverable"+".csv"), index=None)

    def download():
        """
        downloads the file.
        invokes the extract_save method.

        """
        element = driver.find_element_by_xpath(
            '/html/body/form/div[4]/div/div/div[3]/div/div[1]/div[4]/a')
        element.click()
        url = element.get_attribute("href")
        name = url.split("/")[-1]
        time.sleep(3)
        driver.quit()
        extract_save(url.split("/")[-1])
        os.remove(os.path.join(path, name))
        os.remove(os.path.join(path, name.replace(".zip", ".TXT")))

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

    driver = create_driver()
    driver.get(deliverables_url)
    download()


def download_bhavcopy():
    """
    Downloads the bhavcopy of the current day.

    bhav_copy_url = "https://www.bseindia.com/markets/MarketInfo/BhavCopy.aspx"

    creates the driver.
    opens the bhav_copy_url.
    sets the current date.
    downloads the file.

    Methods:
    --------
    create_driver : creates the chrome driver.

    set_date : Sets the day, month, year.

    download : downloads the file. invokes extract_save method.

    extract_save : extracts the file and renames it to the specified name.

    """
    path = os.path.join(os.getcwd(), "Data", "Stock")

    bhav_copy_url = "https://www.bseindia.com/markets/MarketInfo/BhavCopy.aspx"

    def extract_save(name):
        """
        extracts the file and renames it to the specified name.

        Parameters
        ----------
        name : string
            name of the file i.e, security id

        """
        res = zipfile.ZipFile(os.path.join(path, name))
        res.extractall(path)
        today = datetime.datetime.strptime(str(name[2:-8]), "%d%m%y").date()
        bhav = pd.read_csv(os.path.join(path, name[:8]+".CSV"))
        bhav["DATE"] = today
        bhav.to_csv(os.path.join(path, "bhav.csv"), index=None)
        os.remove(os.path.join(path, name[:8]+".CSV"))

    def download():
        """
        downloads the file.
        invokes the extract_save method.

        """
        element = driver.find_element_by_xpath(
            '/html/body/form/div[3]/div[2]/div/div[2]/div/div[2]/div/div/div[1]/table/tbody/tr/td/table[1]/tbody/tr/td/table/tbody/tr[2]/td[1]/table/tbody/tr/td/ul/li[1]/a')
        element.click()
        time.sleep(3)
        url = element.get_attribute("href")
        driver.quit()
        name = url.split("/")[-1]
        extract_save(name)
        os.remove(os.path.join(path, name))

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

    driver = create_driver()
    driver.get(bhav_copy_url)
    download()


def convertBhavCopyToStock(bhav, deli):
    """
    creates a new stock data.

    Parameters
    ----------
    bhav : dataframe

    deli : dataframe

    Returns
    -------

    df : dataframe
        stock dataframe

    """
    path = os.path.join(os.getcwd(), "Data", "companywithid.json")
    ref = json.load(open(path))
    ref = {v: k for k, v in ref.items()}
    df = pd.DataFrame()
    bhav["DATE"] = pd.to_datetime(bhav["DATE"])
    df["Code"] = bhav["SC_CODE"].apply(lambda code: int(code))
    df["Date"] = bhav["DATE"].iloc[0]
    df["Open Price"] = bhav["OPEN"]
    df["High Price"] = bhav["HIGH"]
    df["Low Price"] = bhav["LOW"]
    df["Close Price"] = bhav["CLOSE"]
    df["WAP"] = bhav["NET_TURNOV"] / bhav["NO_OF_SHRS"]
    df["No.of Shares"] = bhav["NO_OF_SHRS"]
    df["No. of Trades"] = bhav["NO_TRADES"]
    df["Total Turnover (Rs.)"] = bhav["NET_TURNOV"]
    df["Deliverable Quantity"] = deli["DELIVERY QTY"]
    df["% Deli. Qty to Traded Qty"] = deli["DELV. PER."]
    df["Spread High-Low"] = bhav["HIGH"] - bhav["LOW"]
    df["Spread Close-Open"] = bhav["CLOSE"] - bhav["OPEN"]
    df["Unix Date"] = bhav["DATE"].apply(lambda x: time.mktime(x.timetuple()))
    df = df.set_index("Code")
    df["Code"] = df.index
    sol = pd.DataFrame()
    for key in ref.keys():
        try:
            sol = sol.append(df.loc[key])
        except:
            pass
    sol["Company"] = sol["Code"].apply(lambda code: ref[code])

    return sol


def update_files():
    path = os.path.join(os.getcwd(), "Data", "Stock")
    result = pd.read_csv(os.path.join(path, "result.csv"))
    result["Code"] = result["Code"].apply(lambda x: int(x))
    result = result.set_index("Code")
    for index, row in result.iterrows():
        try:
            if os.path.exists(os.path.join(path, str(index)+".csv")):
                stk = pd.read_csv(os.path.join(path, str(index)+".csv"))
                stk.loc[len(stk.index)] = row
                stk["Date"] = pd.to_datetime(stk["Date"])
                stk = stk.drop_duplicates(subset=["Date"], keep="first")
                stk = stk.sort_values(by=["Date"], ascending=[False])
                stk.to_csv(os.path.join(path, str(index)+".csv"), index=None)
        except:
            pass


download_deliverables()
download_bhavcopy()
path = os.path.join(os.getcwd(), "Data", "Stock")
bhav = pd.read_csv(os.path.join(path, "bhav.csv"))
deli = pd.read_csv(os.path.join(path, "deliverable.csv"))
result = convertBhavCopyToStock(bhav, deli)
result.to_csv(os.path.join(path, "result.csv"), index=None)
result.to_csv(os.path.join(path, "previousdaystockdetails.csv"), index=None)
update_files()
