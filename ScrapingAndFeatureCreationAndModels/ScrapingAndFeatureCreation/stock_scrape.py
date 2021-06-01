import sys
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import calendar
import datetime


def download_stocks(security_id):
    """
    Downloads the Stock data file.

    stock_url = "https://www.bseindia.com/markets/equity/EQReports/StockPrcHistori.aspx?flag=0"

    creates the driver.

    opens the stock_url.

    sets the security id.

    if file already exists

        sets the from date by taking the last date from the file.
        sets the to date.
        downloads the file.

    if file doesnt exists

        sets the from date
        sets the to date.
        downloads the file.

    Parameters
    ----------
    security_id : string
        security_id of the company

    Returns
    -------
    stock : dataframe

    Methods:
    --------
    create_driver : creates the chrome driver.

    set_to_date : Sets the TO date.

    set_from_date : Sets the FROM date.

    set_security_id : sets the security id.

    download : downloads the file.

    """
    path = os.path.join(os.getcwd(), os.path.join("Data", "Stock"))

    stock_url = "https://www.bseindia.com/markets/equity/EQReports/StockPrcHistori.aspx?flag=0"

    if not os.path.exists(path):
        os.makedirs("Data/Stock")

    def convert_date_to_unix_timestamp(stock_df):
        """
        Adds a new Unix Date column to the given dataframe

        Parameters
        ----------
        stock_df : dataframe

        Returns
        -------
        stock_df : dataframe
            updated dataframe with a new Unix Date column.
        """
        stock_df["Unix Date"] = stock_df["Date"].apply(
            lambda x: time.mktime(x.timetuple()))
        return stock_df

    def set_from_date(d, m, y):
        """
        Sets the FROM date.

        Parameters
        ----------
        d : string
            day

        m : string
            month

        y : string
            year

        """
        from_date = driver.find_element_by_xpath(
            '//*[@id="ContentPlaceHolder1_txtFromDate"]')
        from_date.clear()
        from_date.click()
        year = driver.find_element_by_xpath(
            '/html/body/div[1]/div/div/select[2]')
        year = Select(year)
        while year.options[0].text > y:
            year.select_by_visible_text(year.options[0].text)
            year = driver.find_element_by_xpath(
                '/html/body/div[1]/div/div/select[2]')
            year = Select(year)

        year.select_by_visible_text(y)

        month = driver.find_element_by_xpath(
            '/html/body/div[1]/div/div/select[1]')
        month = Select(month)
        month.select_by_visible_text(m)

        days = driver.find_element_by_xpath(
            "//table/tbody/tr/td/a[text()="+str(d)+"]")
        days.click()

    def set_to_date(d, m, y):
        """
        Sets the TO date.

        Parameters
        ----------
        d : string
            day

        m : string
            month

        y : string
            year

        """
        to_date = driver.find_element_by_xpath(
            '//*[@id="ContentPlaceHolder1_txtToDate"]')
        to_date.clear()
        to_date.click()
        year = driver.find_element_by_xpath(
            '/html/body/div[1]/div/div/select[2]')
        year = Select(year)
        while year.options[0].text > y:
            year.select_by_visible_text(year.options[0].text)
            year = driver.find_element_by_xpath(
                '/html/body/div[1]/div/div/select[2]')
            year = Select(year)

        year.select_by_visible_text(y)

        month = driver.find_element_by_xpath(
            '/html/body/div[1]/div/div/select[1]')
        month = Select(month)
        month.select_by_visible_text(m)

        days = driver.find_element_by_xpath(
            "//table/tbody/tr/td/a[text()="+str(d)+"]")
        days.click()

    def set_security_id(security):
        """
        sets the secuirty id to the input field.

        Parameters
        -----------

        security : string
            security id of the company.

        """
        element = driver.find_element_by_xpath(
            '//*[@id="ContentPlaceHolder1_smartSearch"]')
        element.clear()
        element.send_keys(security)
        element.send_keys(Keys.ENTER)

    def download():
        """
        downloads the file.
        """
        submit = driver.find_element_by_xpath(
            '//*[@id="ContentPlaceHolder1_btnSubmit"]')
        submit.click()
        time.sleep(1)
        driver.find_element_by_xpath(
            "/html/body/form/div[4]/div/div/div[1]/div/div[2]/div/div[1]/div[2]/span/a/i").click()
        time.sleep(3)
        driver.quit()

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

    if os.path.exists(os.path.join(path, str(security_id)+".csv")):
        driver = create_driver()
        driver.get(stock_url)
        old_df = pd.read_csv(os.path.join(
            path, str(security_id)+".csv"))
        old_df["Date"] = pd.to_datetime(old_df["Date"])
        last = old_df["Date"].head(1)[0]
        set_security_id(str(security_id))
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        if today == last.strftime('%Y-%m-%d'):
            return
        set_from_date(
            last.day, calendar.month_abbr[last.month], str(last.year))
        today = datetime.date.today()
        # today = last+datetime.timedelta(365)
        set_to_date(
            today.day, calendar.month_abbr[today.month], str(today.year))
        download()
        new_df = pd.read_csv(os.path.join(
            path, str(security_id)+" (1).csv"))
        new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")
        new_df = new_df.drop(columns=["Unnamed: 13"], axis=1, errors='ignore')
        new_df = new_df.dropna(how='all')
        new_df = convert_date_to_unix_timestamp(new_df)
        res = new_df.append(old_df, ignore_index=True)
        res.to_csv(os.path.join(path, str(security_id)+".csv"), index=None)
        os.remove(os.path.join(path, str(security_id)+" (1).csv"))
        return res.head(365)
    else:
        driver = create_driver()
        driver.get(stock_url)
        set_security_id(str(security_id))
        set_from_date("02", "Aug", "2007")
        today = datetime.date.today()
        # start = datetime.datetime.strptime("01 Jan 2000","%d %b %Y")
        # today = start+datetime.timedelta(365)
        set_to_date(
            today.day, calendar.month_abbr[today.month], str(today.year))
        download()
        stock = pd.read_csv(os.path.join(path, str(security_id)+".csv"))
        stock.Date = pd.to_datetime(stock.Date, errors="coerce")
        stock = stock.drop(columns=["Unnamed: 13"], axis=1, errors='ignore')
        stock = stock.dropna(how='all')
        stock = convert_date_to_unix_timestamp(stock)
        stock.to_csv(os.path.join(path, str(security_id)+".csv"), index=None)
        return stock


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "Data")
    df = pd.read_csv(os.path.join(path, "Equity.csv"))
    codes = df["Security Code"].values.tolist()
    codes.sort()

    for a in codes:
        try:
            download_stocks(a)
        except:
            pass
