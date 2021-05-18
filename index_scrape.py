import os
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import calendar
import datetime
import traceback


def download_index():
    """
    Downloads the index data file.

    index_url = "https://www.bseindia.com/indices/IndexArchiveData.html"
    index = "S&P BSE 500"

    creates the driver.

    opens the index_url.

    sets the index.

    if file already exists

        sets the from date by taking the last date from the file.
        sets the to date.
        downloads the file.
        renames the file to Index.csv

    if file doesnt exists

        sets the from date
        sets the to date.
        downloads the file.
        renames the file to Index.csv

    Methods:
    --------
    create_driver : creates the chrome driver.

    set_to_date : Sets the TO date.

    set_from_date : Sets the FROM date.

    set_index : sets the index.

    download : downloads the file.

    """
    path = os.path.join(os.getcwd(), "Data")

    if not os.path.exists(path):
        os.makedirs("Data")

    index_url = "https://www.bseindia.com/indices/IndexArchiveData.html"
    index = "S&P BSE 500"

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
        from_date = driver.find_element_by_xpath('//*[@id="txtFromDt"]')
        from_date.clear()
        from_date.click()
        year = driver.find_element_by_xpath(
            '/html/body/div[4]/div/div/select[2]')
        year = Select(year)
        while year.options[0].text > y:
            year.select_by_visible_text(year.options[0].text)
            year = driver.find_element_by_xpath(
                '/html/body/div[4]/div/div/select[2]')
            year = Select(year)

        year.select_by_visible_text(y)

        month = driver.find_element_by_xpath(
            '/html/body/div[4]/div/div/select[1]')
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
        to_date = driver.find_element_by_xpath('//*[@id="txtToDt"]')
        to_date.clear()
        to_date.click()
        year = driver.find_element_by_xpath(
            '/html/body/div[4]/div/div/select[2]')
        year = Select(year)
        while year.options[0].text > y:
            print(year.options[0].text, y)
            year.select_by_visible_text(year.options[0].text)
            year = driver.find_element_by_xpath(
                '/html/body/div[4]/div/div/select[2]')
            year = Select(year)

        year.select_by_visible_text(y)

        month = driver.find_element_by_xpath(
            '/html/body/div[4]/div/div/select[1]')
        month = Select(month)
        month.select_by_visible_text(m)

        days = driver.find_element_by_xpath(
            "//table/tbody/tr/td/a[text()="+str(d)+"]")
        days.click()

    def set_index(index_):
        """
        Sets the index field.

        Parameters
        ----------
        index_ : string
            index value
        """
        indexes = driver.find_element_by_xpath('//*[@id="ddlIndex"]')
        indexes = Select(indexes)
        indexes.select_by_visible_text(index_)

    def download():
        """
        downloads the file.

        """
        submit = driver.find_element_by_xpath(
            '/html/body/div[2]/div/div[2]/div[5]/div/input')
        submit.click()
        time.sleep(1)
        driver.find_element_by_xpath(
            "/html/body/div[2]/div/div[1]/div/div[1]/div[2]/i").click()
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
    try:
        driver = create_driver()
        driver.get(index_url)
        set_index(index)
        # set_from_date("7", "Aug", "1999")
        set_from_date("02", "Aug", "2007")

        today = datetime.date.today()
        set_to_date(
            today.day, calendar.month_abbr[today.month], str(today.year))
        download()
        res = pd.read_csv(os.path.join(path, "CSVForDate.csv"), names=[
            "Date", "Open", "High", "Low", "Close"])
        res = res.iloc[1:]
        res[["Open", "High", "Low", "Close"]] = res[[
            "Open", "High", "Low", "Close"]].apply(pd.to_numeric)
        res["Date"] = pd.to_datetime(res["Date"])
        res = res.sort_values(by=["Date"], ascending=[True])
        res = res.drop_duplicates(subset=["Date"], keep="first")
        res["% Return"] = ((res["Close"] / res['Close'].shift(1))-1)*100
        res["% YTD"] = ((res.head(1)['Close'].values[0]/res["Close"])-1)*100
        os.remove(os.path.join(path, "CSVForDate.csv"))
        res.to_csv(os.path.join(path, "Index.csv"), index=None)
        res = res.sort_values(by=["Date"], ascending=[False])
        res.to_csv(os.path.join(path, "sp500.csv"), index=None)
    except:
        traceback.print_exc()


download_index()
