import pyrenko
from utils import *
import pandas as pd
import argparse
import talib
import numpy as np
import scipy.optimize as opt

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

    close_prices_list = read_close_prices_and_times()
    close_prices = pd.DataFrame(close_prices_list)

    # Function for optimization
    def evaluate_renko(brick, history, column_name):
        renko_obj = pyrenko.renko()
        renko_obj.set_brick_size(brick_size=brick, auto=False)
        renko_obj.build_history(prices=history)
        return renko_obj.evaluate()[column_name]

    # Get ATR values (it needs to get boundaries)
    # Drop NaNs
    atr = talib.ATR(high=np.double(close_prices.iloc[:, 2]),
                    low=np.double(close_prices.iloc[:, 3]),
                    close=np.double(close_prices.iloc[:, 4]),
                    timeperiod=14)
    atr = atr[np.isnan(atr) == False]

    # Get optimal brick size as maximum of score function by Brent's (or similar) method
    # First and Last ATR values are used as the boundaries
    optimal_brick_sfo = opt.fminbound(lambda x: -evaluate_renko(brick=x,
                                                                history=close_prices.iloc[:, 4],
                                                                column_name='score'),
                                      np.min(atr),
                                      np.max(atr),
                                      disp=0)

    renko_obj = pyrenko.renko()

    print('Set brick size (manual mode): ', renko_obj.set_brick_size(
        auto=False, brick_size=optimal_brick_sfo))

    renko_obj.build_history(prices=close_prices.iloc[:, 4])

    # print('Renko bar prices: ', renko_obj.get_renko_prices())
    # print('Renko bar directions: ', renko_obj.get_renko_directions())
    print('Renko bar evaluation: ', renko_obj.evaluate())
    print('Balance: ', renko_obj.get_balance())

    if len(renko_obj.get_renko_prices()) > 1:
        renko_obj.plot_renko()
