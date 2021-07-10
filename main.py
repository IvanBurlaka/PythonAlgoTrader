import logging
import os
import time

from dotenv import load_dotenv
load_dotenv()

import pyrenko
from utils import *

from ftx import FtxClient


log = logging.getLogger(__package__)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s> %(message)s"))
log.addHandler(console_handler)
log.setLevel(logging.INFO)


minute = 1
hour = 60*minute
day = 24*hour
week = 7*day

paper_mode = bool(os.getenv('PAPER_MODE', "True").upper() != "FALSE")
limit_order_timeout_seconds = int(os.getenv('LIMIT_ORDER_TIMEOUT_SECONDS'))
atr_stop_multiplier = float(os.getenv('ATR_STOP_MULTIPLIER'))
trailing_history_window = int(os.getenv('TRAILING_HISTORY_WINDOW'))
min_recalculation_period = int(os.getenv('MIN_RECALCULATION_PERIOD'))

def now():
      return int(time.time())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
      resolution = '60' # 1 minute on ftx
      market = os.environ['MARKET']
      subaccount_name = os.environ['SUBACCOUNT_NAME']
      api_key = os.environ['API_KEY']
      api_secret = os.environ['API_SECRET']

      ftx = FtxClient(api_key, api_secret, subaccount_name)

      log.info(f'=================================================================')
      log.info(f'                    (.)(.) sTaRtInG aLgO .i.                     ')
      log.info(f'=================================================================')
      log.info(f'                         market: {market}')
      log.info(f'        trailing history window: {trailing_history_window/hour} hours')
      log.info(f'       min recalculation period: {min_recalculation_period/hour} hours')
      log.info(f'            limit order timeout: {limit_order_timeout_seconds} seconds')
      log.info(f'            atr stop multiplier: {atr_stop_multiplier}')
      log.info(f'                initial balance: {ftx.get_usd_balance()} usd')
      log.info(f'                     paper mode: {paper_mode}')
      log.info(f'=================================================================')

      renko_obj = pyrenko.renko(
            paper_mode,
            ftx, market,
            trailing_history_window,
            min_recalculation_period,
            limit_order_timeout_seconds,
            atr_stop_multiplier
      )
      
      if not paper_mode:
            log.info('closing position if it is open')
            renko_obj.close_position()

      five_minutes = 5*60
      last_complete_candle_time = ftx.get_historical_prices(market, resolution, now()-five_minutes)[-2]["startTime"]
      while True:
            time.sleep(2)
            last_candle = ftx.get_historical_prices(market, resolution, now()-five_minutes)[-2]
            if last_candle["startTime"] > last_complete_candle_time:
                  last_complete_candle_time = last_candle["startTime"]
                  renko_obj.on_new_candle(last_candle)