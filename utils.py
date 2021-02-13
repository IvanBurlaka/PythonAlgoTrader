from binance.client import Client
import pandas as pd

def get_close_prices_and_times(traiding_pair, date_from, date_to, interval):
    api_key = "cVKzTz0yjcqXDB5hQGZcJF4EMFVKZDlQXFzCGVsYN3fVwbeB6ZXvdrcnrzOdAC9d"
    api_secret = "eZya3Izd6CZO2D1GegdMkQzuAmyIw9nengWFPwmVEPoWjBer6ABfWKkxuerYnRIj"
    client = Client(api_key, api_secret)
    print("Starting downloading history")
    bars = client.get_historical_klines(
        symbol=traiding_pair,
        interval=interval,
        start_str=date_from + "T00:00:00+03:00",
        end_str=date_to + "T23:59:00+03:00"
    )
    print("Finished: History size = " + str(len(bars)))
    prices = [float(bar[4]) for bar in bars]
    times = [float(bar[0]) for bar in bars]
    return pd.DataFrame(prices)