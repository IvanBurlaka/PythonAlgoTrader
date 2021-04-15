import pyrenko
from utils import *
import argparse
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--date_from", help = "Inclusive start date of range (Format YYYY-MM-DD)", type=str, required=True)
    parser.add_argument("--date_to", help = "Exclusive end date of range (Format YYYY-MM-DD)", type=str, required=True)
    parser.add_argument("--interval", help = "Interval size (format 1m, 5m, 1h, ...)", type=str, required=True)
    parser.add_argument("--trading_pair", help = "Trading pair (default is KNCUSDT should be related to USDT)", type=str, default='KNCUSDT')

    args = parser.parse_args()

    close_prices = get_close_prices_and_times(args.trading_pair, args.date_from, args.date_to, args.interval)

    renko_obj = pyrenko.renko()

    print('Set brick size (manual mode): ', renko_obj.set_brick_size(auto=True, HLC_history=close_prices.iloc[:,[2,3,4]]))
    renko_obj.build_history(prices=close_prices.iloc[:, 4])
    print('Renko bar prices: ', renko_obj.get_renko_prices())
    print('Renko bar directions: ', renko_obj.get_renko_directions())
    print('Renko bar evaluation: ', renko_obj.evaluate())
    print('Renko bar sma: ', renko_obj.get_sma())
    print('Balance: ', renko_obj.get_balance())

    if len(renko_obj.get_renko_prices()) > 1:
        renko_obj.plot_renko()
