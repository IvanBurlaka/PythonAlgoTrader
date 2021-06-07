import logging
import pickle

from binance.client import Client


log = logging.getLogger(__package__)


def get_close_prices_and_times(trading_pair, date_from, date_to, interval):
    client = Client()

    log.debug("Starting downloading history")

    bars = client.get_historical_klines(
        symbol=trading_pair,
        interval=interval,
        start_str=date_from + "T00:00:00+03:00",
        end_str=date_to + "T23:59:00+03:00"
    )

    log.debug("Finished: History size = " + str(len(bars)))

    return bars


def write_close_prices_and_times(trading_pair, date_from, date_to, interval):
    close_prices_list = get_close_prices_and_times(
        trading_pair, date_from, date_to, interval
    )

    with open('prices.dump', 'wb') as f:
        pickle.dump(close_prices_list, f)


def read_close_prices_and_times(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)

    return data
