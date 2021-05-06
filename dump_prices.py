from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--date_from", help="Inclusive start date of range (Format YYYY-MM-DD)", type=str, required=True)
parser.add_argument(
    "--date_to", help="Exclusive end date of range (Format YYYY-MM-DD)", type=str, required=True)
parser.add_argument(
    "--interval", help="Interval size (format 1m, 5m, 1h, ...)", type=str, required=True)
parser.add_argument(
    "--trading_pair", help="Trading pair (default is KNCUSDT should be related to USDT)", type=str, default='KNCUSDT')

args = parser.parse_args()

close_prices_list = write_close_prices_and_times(
    args.trading_pair, args.date_from, args.date_to, args.interval)
