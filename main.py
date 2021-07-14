from ftx_ws import FtxWebsocketClient
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


if __name__ == '__main__':
      resolution = '60' # 1 minute on ftx
      market = os.environ['MARKET']
      subaccount_name = os.environ['SUBACCOUNT_NAME']
      api_key = os.environ['API_KEY']
      api_secret = os.environ['API_SECRET']

      ftx = FtxClient(api_key, api_secret, subaccount_name)

      position = 0.3
      timeout_seconds = 30

      ##################################################
      class PositionManager:
            # TODO: racing?
            #
            # TODO:
            #  - read last price before execution
            #  - get avg fill price after fill
            #  - get also a fee?
            #  - print diff for stats
            #  - print time took
            #
            # TODO:
            #  - move or not order based on "better" size?

            def __init__(self, ftx: FtxClient, ftxws: FtxWebsocketClient, market: str):
                  self._ftx = ftx
                  self._ftxws = ftxws
                  self._market = market
            
            def _upd_current_position(self):
                  current_position = ftx.get_position(self._market)
                  self._current_position = current_position["size"] if current_position else 0.
                  if current_position and current_position["side"] == "sell":
                        self._current_position = -self._current_position
                  log.info(f'current position={self._current_position} want_position={self._want_position}')
                  if self._current_position == self._want_position:
                        self._done = True
            
            def execute(self, position: float, timeout_seconds: float):
                  self._done = False
                  self._timeout = time.time() + timeout_seconds
                  self._want_position = position
                  self._upd_current_position()
                  if self._want_position == self._current_position:
                        log.info("nothing to do")
                        return

                  self._bid, self._ask = 0., 0.

                  self._ftxws._on_ticker = self._on_ticker
                  self._ftxws._on_fill = self._on_fill

                  self._ftxws.get_fills()
                  self._ftxws.get_ticker(self._market)

                  # wait for fulfillment
                  while True:
                        if self._done:
                              log.info("position filled!")
                              break
                        if self._is_timeout():
                              log.info("timeout!")
                              break
                        time.sleep(0.1)
                  
                  # TODO: clenup!
                  self._ftxws._on_ticker = None
                  self._ftxws._on_fill = None

            def _longing(self) -> bool:
                  return self._want_position - self._current_position > 0
            
            def _shorting(self) -> bool:
                  return self._want_position - self._current_position < 0
            
            def _is_closing(self) -> bool:
                  return self._want_position == 0.
            
            def _is_timeout(self) -> bool:
                  return time.time() > self._timeout

            def _on_ticker(self, data):
                  if self._done:
                        return

                  min_change = data['ask']/1500 # TODO

                  # TODO: price increment!
                  if self._longing() and self._bid != data['bid']:
                        if abs(self._bid - data['bid']) < min_change:
                              return
                        
                        log.info(f'bid changed: bid={data["bid"]} min_change={min_change:.4f}')
                        self._bid = data['bid']
                        self._ftx.cancel_orders(self._market)
                        self._upd_current_position() # TODO: can be done in fills subscription
                        size = self._want_position - self._current_position
                        if size <= 0:
                              return
                        price = self._bid + 0.01
                        log.info(f'long price={price} size={size} current_position={self._current_position} want_position={self._want_position}')
                        self._ftx.place_order(
                              market=self._market,
                              side='buy',
                              size=size,
                              price=price,
                              type='limit',
                              post_only=True,
                              reduce_only=self._is_closing()
                        )
                  if self._shorting() and self._ask != data['ask']:
                        if abs(self._ask - data['ask']) < min_change:
                              return

                        log.info(f'ask changed: ask={data["ask"]} min_change={min_change:.4f}')
                        self._ask = data['ask']
                        self._ftx.cancel_orders(self._market)
                        self._upd_current_position() # TODO: can be done in fills subscription
                        size =  abs(self._current_position - self._want_position)
                        if size <= 0:
                              return
                        price = self._ask - 0.01
                        log.info(f'short price={price} size={size} current_position={self._current_position} want_position={self._want_position}')
                        resp = self._ftx.place_order(
                              market=self._market,
                              side='sell',
                              size=size,
                              price=price,
                              type='limit',
                              post_only=True,
                              reduce_only=self._is_closing()
                        )
                        log.info(f'short response: {resp}')
            
            def _on_fill(self, data):
                  log.info(f'new fill: {data}')
                  # TODO: filter by market!!!
                  self._upd_current_position()

      from ftx_ws import FtxWebsocketClient

      ftxws = FtxWebsocketClient(api_key, api_secret, subaccount_name)

      import sys
      if len(sys.argv) < 2:
            log.error("give position as cmd line argument")
            exit()
      position = float(sys.argv[1])
      timeout = 60

      input(f'market:\t\t{market}\nposition:\t{position}\ntimeout:\t{timeout}s\n\nye?')

      PositionManager(ftx, ftxws, market).execute(position, timeout)
      ###################################################


# if __name__ == '__main__':
#       resolution = '60' # 1 minute on ftx
#       market = os.environ['MARKET']
#       subaccount_name = os.environ['SUBACCOUNT_NAME']
#       api_key = os.environ['API_KEY']
#       api_secret = os.environ['API_SECRET']

#       ftx = FtxClient(api_key, api_secret, subaccount_name)

#       log.info(f'=================================================================')
#       log.info(f'                    (.)(.) sTaRtInG aLgO .i.                     ')
#       log.info(f'=================================================================')
#       log.info(f'                         market: {market}')
#       log.info(f'        trailing history window: {trailing_history_window/hour} hours')
#       log.info(f'       min recalculation period: {min_recalculation_period/hour} hours')
#       log.info(f'            limit order timeout: {limit_order_timeout_seconds} seconds')
#       log.info(f'            atr stop multiplier: {atr_stop_multiplier}')
#       log.info(f'                initial balance: {ftx.get_usd_balance()} usd')
#       log.info(f'                     paper mode: {paper_mode}')
#       log.info(f'=================================================================')

#       renko_obj = pyrenko.renko(
#             paper_mode,
#             ftx, market,
#             trailing_history_window,
#             min_recalculation_period,
#             limit_order_timeout_seconds,
#             atr_stop_multiplier
#       )
      
#       if not paper_mode:
#             log.info('closing position if it is open')
#             renko_obj.close_position()

#       five_minutes = 5*60
#       last_complete_candle_time = ftx.get_historical_prices(market, resolution, now()-five_minutes)[-2]["startTime"]
#       while True:
#             time.sleep(2)
#             last_candle = ftx.get_historical_prices(market, resolution, now()-five_minutes)[-2]
#             if last_candle["startTime"] > last_complete_candle_time:
#                   last_complete_candle_time = last_candle["startTime"]
#                   renko_obj.on_new_candle(last_candle)