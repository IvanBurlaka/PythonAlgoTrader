import ftx
import numpy as np
import talib
import logging
import time

import scipy.optimize as opt
from utils import *
import pandas as pd


log = logging.getLogger(__package__)


def get_initial_history(market, minutes):
    # from binance, bc ftx stupid no has data
    from datetime import date, timedelta

    market = market.replace('-PERP', 'USDT')
    days = int(minutes/24/60)+1
    today = date.today()

    start = (today - timedelta(days)).strftime("%Y/%m/%d")
    end = (today + timedelta(days=1)).strftime("%Y/%m/%d")

    log.info(f'history start: {start}')
    log.info(f'history end: {end}')

    history = get_close_prices_and_times(market, start, end, '1m')
    log.info(f'history length: {len(history)}')
    log.info(f'last candle in history: {history[-1]}')

    return history


class renko:
    def __init__(
        self,
        paper_mode: bool,
        ftx: ftx.FtxClient,
        market: str,
        trailing_history_window: int,
        min_recalculation_period: int,
        limit_order_timeout_seconds: int,
        atr_stop_multiplier: float
    ):
        self.paper_mode = paper_mode
        self.market = market
        self.ftx = ftx
        self.limit_order_timeout_seconds = limit_order_timeout_seconds
        self.atr_stop_multiplier = atr_stop_multiplier
        self.renko_prices = []
        self.renko_directions = []
        self.renko_prices_for_calculation = []
        self.renko_directions_for_calculation = []
        #self.current_capital = 1000
        self.current_capital = self.ftx.get_usd_balance()
        self.size_increment = ftx.get_future(market)["sizeIncrement"]
        self.atr = None
        self.atr_stop_loss = None
        
        # trend following params
        self.position_data = {"trade_direction": "", "prices_opened": []}
        self.profit = []
        self.capital_history = []
        # self.close_price = pd.DataFrame(read_close_prices_and_times(prices_file))
        self.candles = pd.DataFrame(get_initial_history(self.market, trailing_history_window))
        self.trailing_history_window = trailing_history_window
        self.min_recalculation_period = min_recalculation_period
        self.last_recalculation_index = 0
        self.number_of_candles_calculations = 9

        self.candles_since_recalculation = 0

        self.set_brick_size(auto=True)

        # Init by start values
        self.renko_prices.append(float(self.candles.iloc[-1, 4]))
        self.renko_directions.append(0)

        log.info(f'initial brick close: {self.renko_prices[-1]}')
        log.info(f"size increment: {self.size_increment}")

    def calculate_optimal_brick_size(self):
        # self.last_recalculation_index = current_candle
        # Function for optimization
        def evaluate_renko(brick, history, column_name):
            self.set_brick_size(brick_size=brick, auto=False)
            self.build_history_for_calculation(history)
            return self.evaluate_for_calculation(history)[column_name]

        # Get ATR values (it needs to get boundaries)
        # Drop NaNs
        atr = talib.ATR(high=np.double(self.candles.iloc[-self.trailing_history_window:, 2]),
                        low=np.double(self.candles.iloc[-self.trailing_history_window:, 3]),
                        close=np.double(self.candles.iloc[-self.trailing_history_window:, 4]),
                        timeperiod=14)
        atr = atr[np.isnan(atr) == False]

        # Get optimal brick size as maximum of score function by Brent's (or similar) method
        # First and Last ATR values are used as the boundaries
        renko_values=[]
        candles_spread=0
        while candles_spread < self.number_of_candles_calculations:
            renko_values.append(opt.fminbound(lambda x: -evaluate_renko(brick=x,
                                                       history=self.candles.iloc[-self.trailing_history_window+candles_spread:, 4],
                                                       column_name='score'),
                             np.min(atr),
                             np.max(atr),
                             disp=0))
            candles_spread += 1

        result = np.median(renko_values)
        log.info(f'calculated optimal brick size: {result}')
        self.candles_since_recalculation = 0
        return result

    # Setting brick size. Auto mode is preferred, it uses history
    def set_brick_size(self, HLC_history=None, auto=True, brick_size=10.0):
        if auto == True:
            self.brick_size = self.calculate_optimal_brick_size()
        else:
            self.brick_size = brick_size
        return self.brick_size

    def __trend_following_strategy(self):
        if self.renko_directions[-1] == 0:
            log.info("waiting for more bricks")
            return

        last_close_price = self.candles.iloc[-1, 4]

        if self.renko_directions[-2] == self.renko_directions[-1]:
            # direction matches previous, then open position in following direction
            position = self.ftx.get_position(self.market)
            
            if not position or not position["size"]:
                # if there's no position and no orders, open position
                size = self.current_capital/last_close_price
                size -= size % self.size_increment # adjust size to be multiple of size_increment
                if self.renko_directions[-1] == 1:
                    position_side = "long"
                    side = ftx.buy
                    self.atr_stop_loss = last_close_price - self.atr_stop_multiplier*self.atr
                else:
                    position_side = "short"
                    side = ftx.sell
                    self.atr_stop_loss = self.renko_prices[-1] + self.atr_stop_multiplier*self.atr
                self.start_iteration(side=side, size=size, max_wait_seconds=self.limit_order_timeout_seconds, price=last_close_price)
                self.position_data["trade_direction"] = position_side
            else:
                # there's open position, do nothing
                log.info('waiting for position close conditions')
            self.position_data["prices_opened"].append(last_close_price)
        else:
            log.info('waiting for confirmation brick in the same direction')

    def start_iteration(self, side: str, size:float, max_wait_seconds:float=0., price:float=0.):
        log.info(f'opening position, waiting {max_wait_seconds} sec at price {price}: side={side}, size={size}, atr_stop={self.atr_stop_loss}')
        if not self.paper_mode:
            self.open_position(side=side, size=size, max_wait_seconds=max_wait_seconds, price=price)
        
    def open_position(self, side: str, size:float, max_wait_seconds:float=0., price:float=0.):
        self.ftx.cancel_orders(market=self.market)

        position = self.ftx.get_position(self.market)
        if position and position["size"] > 0 and position["side"] != side:
            log.error(f"opening position: unexpected open position: open side={position['side']}, open size={position['size']}, want side={side}")
        
        opened_size = position["size"] if position else 0. # note: position["size"] is absolute value
        remaining_size = size - opened_size

        # opened_size is a subject to size_increment for the market
        # so it can be slightly different from wanted size
        if remaining_size < self.size_increment:
            log.info("position opened")
            return

        if max_wait_seconds == 0:
            # market open position
            o = self.ftx.place_order(
                market=self.market,
                side=side,
                price="0",
                type=ftx.market,
                size=remaining_size
            )
            log.info(f"opening position: market fill: price={o['price']}, size={o['size']}, id={o['id']}")
        else:
            # try to limit open position
            o = self.ftx.place_order(
                market=self.market,
                side=side,
                price=price,
                type=ftx.limit,
                size=size
            )
            log.info(f"opening position: waiting for limit order fill: price={o['price']}, size={o['size']}, id={o['id']}")
            time.sleep(max_wait_seconds)
            # ensure market open of position if it's still pending
            self.open_position(side=side, size=size, max_wait_seconds=0.)
    
    def finish_iteration(self, reason: str, max_wait_seconds:float=0., price:float=0.):
        if not self.position_data["trade_direction"]:
            return

        log.info(f'closing position, waiting {max_wait_seconds} sec at price {price}: reason - {reason}')
        if not self.paper_mode:
            self.close_position(max_wait_seconds, price)
        
        self.current_capital = self.ftx.get_usd_balance()
        log.info(f'balance: {self.current_capital} usd')
        self.capital_history.append(self.current_capital)
        self.atr_stop_loss = None
        self.position_data["trade_direction"] = None
        self.position_data["prices_opened"] = []
        # recalculate brick size
        if self.candles_since_recalculation > self.min_recalculation_period:
            self.brick_size = self.calculate_optimal_brick_size()
    
    def close_position(self, max_wait_seconds:float=0., price:float=0.):
        self.ftx.cancel_orders(market=self.market)

        position = self.ftx.get_position(self.market)
        if not position or not position["size"]:
            log.info("position closed")
            return

        size = position["size"]
        side = ftx.sell if position["side"] == ftx.buy else ftx.buy

        if max_wait_seconds == 0:
            # market close position
            o = self.ftx.place_order(
                market=self.market,
                side=side,
                price="0",
                type=ftx.market,
                size=size,
                reduce_only=True,
            )
            log.info(f"closing position: market fill: price={o['price']}, size={o['size']}, id={o['id']}")
        else:
            # try to limit close position
            o = self.ftx.place_order(
                market=self.market,
                side=side,
                price=price,
                type=ftx.limit,
                size=size,
                reduce_only=True,
            )
            log.info(f"closing position: waiting for limit order fill: price={o['price']}, size={o['size']}, id={o['id']}")
            time.sleep(max_wait_seconds)
            # ensure market close of position if it's still open
            self.close_position(0)

    def on_new_candle(self, candle):
        # convert ftx candle to binance candle bc close_price is binance format
        self.candles_since_recalculation += 1
        self.candles = self.candles.append([[
            candle['time'],
            candle['open'],
            candle['high'],
            candle['low'],
            candle['close'],
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]], ignore_index=True)
        self.atr = talib.ATR(high=np.double(self.candles.iloc[-15:, 2]),
                            low=np.double(self.candles.iloc[-15:, 3]),
                            close=np.double(self.candles.iloc[-15:, 4]),
                            timeperiod=14)[-1]
        log.info(f'new candle: close price={candle["close"]} start time={candle["startTime"]}, atr={self.atr:.3f}')
        self.__renko_rule(self.candles.iloc[-1, 4])

    def __renko_rule(self, last_close_price):
        #log.info(f'running renko rule on last price: {last_price}')
        # Get the gap between two prices
        gap_div = int(float(last_close_price - self.renko_prices[-1]) / self.brick_size)
        is_new_brick = False
        start_brick = 0
        num_new_bars = 0
        is_direction_opposite = False

        # When we have some gap in prices
        if gap_div != 0:
            # Forward any direction (up or down)
            if (gap_div > 0 and (self.renko_directions[-1] > 0 or self.renko_directions[-1] == 0)) or (
                    gap_div < 0 and (self.renko_directions[-1] < 0 or self.renko_directions[-1] == 0)):
                num_new_bars = gap_div
                is_new_brick = True
                start_brick = 0
            # Backward direction (up -> down or down -> up)
            elif np.abs(gap_div) >= 2:  # Should be double gap at least
                num_new_bars = gap_div
                num_new_bars -= np.sign(gap_div)
                start_brick = 2
                is_new_brick = True
                is_direction_opposite = True
                brick_price = self.renko_prices[-1] + 2 * self.brick_size * np.sign(gap_div)
                self.renko_prices.append(brick_price)
                self.renko_directions.append(np.sign(gap_div))
                direction = "up" if self.renko_directions[-1] > 0 else "down"
                log.info(f'new brick ({direction}): {brick_price}')
            # else:
            # num_new_bars = 0

            if is_new_brick:
                # Add each brick
                for _ in range(start_brick, np.abs(gap_div)):
                    brick_price = self.renko_prices[-1] + self.brick_size * np.sign(gap_div)
                    self.renko_prices.append(brick_price)
                    self.renko_directions.append(np.sign(gap_div))
                    direction = "up" if self.renko_directions[-1] > 0 else "down"
                    log.info(f'new brick ({direction}): {brick_price}')
                self.__trend_following_strategy()

        # check stop conditions if in position
        if self.position_data["trade_direction"]:
            # atr stop loss rule
            if self.atr_stop_loss:
                if self.renko_directions[-1] > 0 and last_close_price < self.atr_stop_loss:
                    reason = f"stop loss: candle close below (entry brick - atr): candle close={last_close_price}, atr_stop={self.atr_stop_loss}"
                    self.finish_iteration(reason,
                        self.limit_order_timeout_seconds,
                        price=last_close_price)
                elif self.renko_directions[-1] < 0 and last_close_price > self.atr_stop_loss:
                    reason = f"stop loss: candle close above (entry brick + atr): candle close={last_close_price}, atr_stop={self.atr_stop_loss}"
                    self.finish_iteration(reason,
                        self.limit_order_timeout_seconds,
                        price=last_close_price)
            
            # brick open stop loss rule
            if self.renko_directions[-1] > 0 and last_close_price < self.renko_prices[-1] - self.brick_size:
                reason = f"stop loss: candle close below last brick open: candle close={last_close_price} last brick open={self.renko_prices[-1] - self.brick_size}"
                self.finish_iteration(reason,
                    self.limit_order_timeout_seconds,
                    price=last_close_price)
            elif self.renko_directions[-1] < 0 and last_close_price > self.renko_prices[-1] + self.brick_size:
                reason = f"stop loss: candle close above last brick open: candle close={last_close_price} last brick open={self.renko_prices[-1] + self.brick_size}"
                self.finish_iteration(reason,
                    self.limit_order_timeout_seconds,
                    price=last_close_price)

        return num_new_bars

    def __renko_rule_for_calculation(self, last_price, candle_index):
        # Get the gap between two prices
        gap_div = int(
            float(last_price - self.renko_prices_for_calculation[-1]) / self.brick_size)
        is_new_brick = False
        start_brick = 0
        num_new_bars = 0

        # When we have some gap in prices
        if gap_div != 0:
            # Forward any direction (up or down)
            if (gap_div > 0 and (self.renko_directions_for_calculation[-1] > 0 or self.renko_directions_for_calculation[-1] == 0)) or (
                    gap_div < 0 and (self.renko_directions_for_calculation[-1] < 0 or self.renko_directions_for_calculation[-1] == 0)):
                num_new_bars = gap_div
                is_new_brick = True
                start_brick = 0
            # Backward direction (up -> down or down -> up)
            elif np.abs(gap_div) >= 2:  # Should be double gap at least
                num_new_bars = gap_div
                num_new_bars -= np.sign(gap_div)
                start_brick = 2
                is_new_brick = True
                self.renko_prices_for_calculation.append(
                    self.renko_prices_for_calculation[-1] + 2 * self.brick_size * np.sign(gap_div))
                self.renko_directions_for_calculation.append(np.sign(gap_div))
            # else:
            # num_new_bars = 0

            if is_new_brick:
                # Add each brick
                for d in range(start_brick, np.abs(gap_div)):
                    self.renko_prices_for_calculation.append(
                        self.renko_prices_for_calculation[-1] + self.brick_size * np.sign(gap_div))
                    self.renko_directions_for_calculation.append(np.sign(gap_div))

        return num_new_bars

    # Getting renko on history
    def build_history_for_calculation(self, history):
        if len(history) > 0:
            # Init by start values
            self.renko_prices_for_calculation.append(float(history.iloc[0]))
            self.renko_directions_for_calculation.append(0)

            # For each price in history
            for index, p in enumerate(history[1:]):
                self.__renko_rule_for_calculation(float(p), index)

        return len(self.renko_prices)

    def evaluate_for_calculation(self, history, method='simple'):
        balance = 0
        sign_changes = 0
        price_ratio = len(history) / len(self.renko_prices_for_calculation)

        if method == 'simple':
            for i in range(2, len(self.renko_directions_for_calculation)):
                if self.renko_directions_for_calculation[i] == self.renko_directions_for_calculation[i - 1]:
                    balance = balance + 1
                else:
                    balance = balance - 2
                    sign_changes = sign_changes + 1

            if sign_changes == 0:
                sign_changes = 1

            score = balance / sign_changes
            if score >= 0 and price_ratio >= 1:
                score = np.log(score + 1) * np.log(price_ratio)
            else:
                score = -1.0
            self.renko_prices_for_calculation = []
            self.renko_directions_for_calculation = []
            return {'balance': balance, 'sign_changes:': sign_changes,
                    'price_ratio': price_ratio, 'score': score}