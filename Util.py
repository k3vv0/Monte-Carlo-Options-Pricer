import os
from datetime import datetime, date, timedelta
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv


def openWindow():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


def calcDaysBetween(exp_date):
    if isinstance(exp_date, datetime):
        exp_date = exp_date.date()
    elif isinstance(exp_date, str):
        try:
            exp_date = datetime.strptime(exp_date, "%Y-%m-%d").date()
        except ValueError as e:
            print(f"Error parsing the expiry date '{exp_date}': {e}")
            return None
    difference = exp_date - date.today()
    # print("Days:", difference.days)
    return difference.days


def definePeriod(period="1m"):
    today = datetime.today().date()

    if period == "1w":
        end_date = today + timedelta(weeks=1)
    elif period == "1m":
        end_date = today + timedelta(weeks=4)
    elif period == "3m":
        end_date = today + timedelta(weeks=12)
    elif period == "6m":
        end_date = today + timedelta(weeks=24)
    elif period == "1y":
        end_date = today + timedelta(weeks=52)
    elif period == "2y":
        end_date = today + timedelta(weeks=104)
    else:
        raise ValueError("Invalid period specified")

    return today.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def get_risk_free_rate(duration='1m'):
    load_dotenv()
    fred_api_key = os.getenv('FRED_API_KEY')
    fred = Fred(api_key=fred_api_key)

    # Mapping of user input to the FRED series codes
    duration_mapping = {
        '1m': 'DTB4WK',
        '3m': 'DTB3',
        '6m': 'DTB6',
        '1y': 'DTB1YR',
    }

    if duration in duration_mapping:
        tbill_data = fred.get_series(duration_mapping[duration])
        tbill_yield = tbill_data.iloc[-1]
    else:
        raise ValueError(f"Invalid duration: {duration}. Valid options are {list(duration_mapping.keys())}")

    return tbill_yield / 100
