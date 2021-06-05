import argparse
import pyrenko
import os
from utils import *

import time

from dotenv import load_dotenv
load_dotenv()

# from ciso8601 import parse_datetime

from ftx import FtxClient

minute = 1
hour = 60*minute
day = 24*hour
week = 7*day

def now():
      return int(time.time())

# def get_initial_history(ftx, minutes):
#       def parse_time(t):
#             return parse_datetime(t).timestamp()

#       result = ftx.get_historical_prices(market, '60', now()-(minutes+1)*60)

#       while True:
#             print(result[-1]["startTime"])
#             last_time = parse_time(result[-1]["startTime"])
#             candles = ftx.get_historical_prices(market, '60', last_time)
#             candles = [c for c in candles if parse_time(c["startTime"]) > last_time]
#             result.extend(candles)
#             print(f'adding {len(candles)} candles')
#             if not candles:
#                   break

#       return result[:-1]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
      market = os.environ['MARKET']
      subaccount_name = os.environ['SUBACCOUNT_NAME']
      api_key = os.environ['API_KEY']
      api_secret = os.environ['API_SECRET']

      ftx = FtxClient(api_key, api_secret, subaccount_name)

      parser = argparse.ArgumentParser()
      parser.add_argument("--prices_file", help="File with prices", type=str, required=True)
      args = parser.parse_args()

      trailing_history_window = 3*day
      min_recalculation_period = 6*day

      renko_obj = pyrenko.renko(ftx, market, args.prices_file, trailing_history_window, min_recalculation_period)
      renko_obj.set_brick_size(auto=True)
      renko_obj.build_history()

      renko_obj.evaluate()
      print('Balance: ', renko_obj.get_balance())

      # ftx.place_order(
      #       market=market,
      #       side=ftx.buy,
      #       price=32.05,
      #       size=1,
      #       type=ftx.limit,
      # )

      # ftx.cancel_orders(market=market)
      # print('canceled order')


      # history = ftx.get_historical_prices(market, '60', now()-10*60)
      # last_time = history[-2]["startTime"]
      # print('last 10 minutes:')
      # for c in history:
      #       close = c["close"]
      #       t = c["startTime"]
      #       print(f'\tprice: {close}\ttime={t}')
      
      # while True:
      #       time.sleep(2)
      #       history = ftx.get_historical_prices(market, '60', now()-5*60)
      #       if history[-2]["startTime"] > last_time:
      #             print('new candle:')
      #             close = history[-2]["close"]
      #             t = history[-2]["startTime"]
      #             last_time = t
      #             print(f'\tprice: {close}\ttime={t}')