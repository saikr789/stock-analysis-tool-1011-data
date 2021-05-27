import sys
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import time
import calendar
import datetime
import pandas as pd
import traceback
import multiprocessing
from multiprocessing.pool import ThreadPool
import traceback
import subprocess


def download_corporate_actions(security_id):
    """
    Downloads the corporate actions of the give security id.
    corporate_url = "https://www.bseindia.com/corporates/corporate_act.aspx"
    creates the driver.
    opens the corporate_url.
    sets the from date.
    sets the to date.
    downloads the file.
    replaces the if already downloaded.
    Parameters
    ----------
    security_id : string
        security id of the company.
    Methods:
    --------
    create_driver : creates the chrome driver.
    set_security_id : sets the security id.
    set_to_date : Sets the TO date.
    set_from_date : Sets the FROM date.
    download : downloads the file.
    """
    security_id = str(security_id)
    corporate_url = "https://www.bseindia.com/corporates/corporate_act.aspx"

    path = os.path.join(os.getcwd(), os.path.join("Data", "CorporateActions"))

    if not os.path.exists(path):
        os.makedirs("Data/CorporateActions")

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
            '//*[@id="ContentPlaceHolder1_txtDate"]')
        from_date.clear()
        from_date.click()
        year = driver.find_element_by_xpath(
            '/html/body/div[2]/div/div/select[2]')
        year = Select(year)
        while year.options[0].text > y:
            year.select_by_visible_text(year.options[0].text)
            year = driver.find_element_by_xpath(
                '/html/body/div[2]/div/div/select[2]')
            year = Select(year)

        year.select_by_visible_text(y)

        month = driver.find_element_by_xpath(
            '/html/body/div[2]/div/div/select[1]')
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
            '//*[@id="ContentPlaceHolder1_txtTodate"]')
        to_date.clear()
        to_date.click()
        year = driver.find_element_by_xpath(
            '/html/body/div[2]/div/div/select[2]')
        year = Select(year)
        while year.options[0].text > y:
            print(year.options[0].text, y)
            year.select_by_visible_text(year.options[0].text)
            year = driver.find_element_by_xpath(
                '/html/body/div[2]/div/div/select[2]')
            year = Select(year)

        year.select_by_visible_text(y)

        month = driver.find_element_by_xpath(
            '/html/body/div[2]/div/div/select[1]')
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
            '//*[@id="ContentPlaceHolder1_SmartSearch_smartSearch"]')
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
            "/html/body/div[1]/form/div[4]/div/div/div[2]/div/div/div[2]/a/i").click()
        time.sleep(3)
        driver.quit()

    def create_driver():
        """
        Creates a Chrome Driver.
        Returns:
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
        driver.get(corporate_url)
        set_security_id(str(security_id))
        # set_from_date("01", "Jan", "1991")
        set_from_date("02", "Aug", "2007")
        today = datetime.date.today()
        set_to_date(
            today.day, calendar.month_abbr[today.month], str(today.year))
        download()
        if os.path.exists(os.path.join(path, str(security_id)+".csv")):
            os.remove(os.path.join(
                path, str(security_id)+".csv"))
        os.rename(os.path.join(path, "Corporate_Actions.csv"),
                  os.path.join(path, str(security_id)+".csv"))

    except:
        cols = ['Security Code', 'Security Name', 'Company Name', 'Ex Date', 'Purpose', 'Record Date',
                'BC Start Date', 'BC End Date', 'ND Start Date', 'ND End Date', 'Actual Payment Date']
        df = pd.DataFrame(columns=cols)
        df.to_csv(os.path.join(path, str(security_id)+".csv"), index=None)


df = pd.read_csv(os.path.join(os.getcwd(), "Data", "Equity.csv"))

security_codes = df["Security Code"].values.tolist()
security_codes.sort()
for code in security_codes:
    try:
        download_corporate_actions(code)
    except:
        traceback.print_exc()
