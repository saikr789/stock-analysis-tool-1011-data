import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import time


def download_equity():
    """
    download the equity file.

    if file already exists, returns None

    security_url = "https://www.bseindia.com/corporates/List_Scrips.aspx"

    creates the driver.

    opens the security_url.

    Sets Active and Equity fields.

    downloads the file.

    """
    path = os.path.join(os.getcwd(), "Data")

    if not os.path.exists(path):
        os.makedirs("Data")

    security_url = "https://www.bseindia.com/corporates/List_Scrips.aspx"

    if os.path.exists(os.path.join(path, "Equity.csv")):
        print("Equity.csv exists")
        return

    chromeOptions = webdriver.ChromeOptions()
    chromeOptions.add_argument("--headless")
    chromeOptions.add_experimental_option(
        "prefs", {"download.default_directory": path})
    driver = webdriver.Chrome(
        ChromeDriverManager().install(), options=chromeOptions)
    driver.get(security_url)

    # to select Equity
    equity = driver.find_element_by_xpath(
        '//*[@id="ContentPlaceHolder1_ddSegment"]')
    equity = Select(equity)
    equity.select_by_visible_text("Equity")

    # to select Active
    active = driver.find_element_by_xpath(
        '//*[@id="ContentPlaceHolder1_ddlStatus"]')
    active = Select(active)
    active.select_by_visible_text("Active")

    # to click submit
    submit = driver.find_element_by_xpath(
        '//*[@id="ContentPlaceHolder1_btnSubmit"]')
    submit.send_keys(Keys.RETURN)

    # to download csv file
    driver.find_element_by_xpath(
        "/html/body/div[1]/form/div[4]/div/div/div[2]/div/div/div[2]/a/i").click()
    time.sleep(3)
    driver.quit()


if __name__ == "__main__":
    download_equity()
