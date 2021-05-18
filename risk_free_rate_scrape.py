import os
import traceback
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import datetime
from bs4 import BeautifulSoup

import sys


def download_risk_free_rate():
    """
    Downloads the Risk Free Rate file.

    risk_free_rate_url = "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield"

    creates the driver.
    opens the risk_free_rate_url.
    downloads the file.

    Methods:
    --------

    create_driver : creates the chrome driver.

    download : extracts the data from the page and saves to a csv file.

    """
    path = os.path.join(os.getcwd(), "Data")
    # risk_free_rate_url = "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield"
    risk_free_rate_url = "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldAll"
    if not os.path.exists(path):
        os.makedirs("Data")

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
        downloads the risk free rate file.

        """
        # ele = driver.find_element_by_xpath('//*[@id="interestRateTimePeriod"]')
        # ele = Select(ele)
        # ele.select_by_visible_text("All")

        # btn = driver.find_element_by_xpath(
        #     '/html/body/form/div[8]/div/div[1]/div/div[2]/div/div/div/div[1]/div[2]/div/table/tbody/tr/td/div/div[3]/div[2]/input')
        # btn.click()
        # time.sleep(15)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        driver.quit()
        table = soup.find_all("table", {"class": "t-chart"})
        risk_free_rate = pd.read_html(str(table))[0]
        risk_free_rate.to_csv(os.path.join(
            path, "RiskFreeRateFull.csv"), index=None)
        risk_free = risk_free_rate[["Date", "3 mo"]]
        risk_free.Date = pd.to_datetime(risk_free.Date, errors="coerce")
        start = datetime.datetime(2007, 8, 2)
        end = datetime.datetime.now()
        risk_free = risk_free[risk_free.Date.between(start, end)]
        risk_free.columns = ["Date", "Rate"]
        risk_free.dropna(inplace=True)
        risk_free.to_csv(os.path.join(path, "RiskFreeRate.csv"), index=None)
    try:
        driver = create_driver()
        driver.get(risk_free_rate_url)
        download()
    except:
        traceback.print_exc()


download_risk_free_rate()
