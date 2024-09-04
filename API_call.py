import requests
import pandas as pd
import io


def response(ticker):
    r = requests.get(
        'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&apikey=WQSRADK4DGSR96FZ&datatype=csv'.format(ticker))

    return r
