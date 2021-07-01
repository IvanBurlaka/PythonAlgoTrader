import pyrenko
from utils import *
import pandas as pd
import argparse
import talib
import numpy as np
import scipy.optimize as opt
from sys import exit

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #    parser = argparse.ArgumentParser()
    #    parser.add_argument(
    #        "--date_from", help="Inclusive start date of range (Format YYYY-MM-DD)", type=str, required=True)
    #    parser.add_argument(
    #        "--date_to", help="Exclusive end date of range (Format YYYY-MM-DD)", type=str, required=True)
    #    parser.add_argument(
    #        "--interval", help="Interval size (format 1m, 5m, 1h, ...)", type=str, required=True)
    #    parser.add_argument(
    #        "--trading_pair", help="Trading pair (default is KNCUSDT should be related to USDT)", type=str, default='KNCUSDT')

    #   args = parser.parse_args()
    # renko_obj = pyrenko.renko(close_prices.iloc[:,[2,3,4]])

    # print('optimal', optimal_brick_sfo)

    hour = 60
    day = 24*hour

    time_increment = day
    divider = 1

    while divider <= 5:
        renko_obj = pyrenko.renko(trailing_history=30*day, largest_trailing_history=30*day, position_divider=divider )
        print('Set brick size (manual mode): ',
        renko_obj.set_brick_size(auto=True))
        renko_obj.build_history()
        divider+=1

        # print('Renko bar prices: ', renko_obj.get_renko_prices())
        # print('Renko bar directions: ', renko_obj.get_renko_directions())
        print('Renko bar evaluation: ', renko_obj.evaluate())
        print('Balance: ', renko_obj.get_balance())

#    if len(renko_obj.get_renko_prices()) > 1:
#       renko_obj.plot_renko()
