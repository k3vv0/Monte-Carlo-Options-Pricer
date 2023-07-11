from datetime import datetime, date
import pandas as pd


def openWindow():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


def calcDaysBetween(exp_date):
    exp_date = str(exp_date)
    exp_date = exp_date[8:18]
    # print(exp_date)
    date_format = "%Y-%m-%d"
    date1 = datetime.strptime(str(exp_date), date_format)
    difference = date1.date() - date.today()
    # print("Days:", difference.days)
    return difference.days
