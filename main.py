import argparse
import pyrenko
from utils import *

hour = 60
day = 24*hour
week = 7*day

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices_file", help="File with prices", type=str, required=True)
    args = parser.parse_args()

    trailing_history_window = 3*day
    min_recalculation_period = 6*day

    renko_obj = pyrenko.renko(args.prices_file, trailing_history_window, min_recalculation_period)
    renko_obj.set_brick_size(auto=True)
    renko_obj.build_history()
    
    renko_obj.evaluate()
    print('Balance: ', renko_obj.get_balance())