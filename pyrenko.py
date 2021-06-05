import ftx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import talib
import logging

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


class renko:
    def __init__(self, ftx: ftx.FtxClient, market, prices_file, trailing_history_window, min_recalculation_period):
        self.market = market

        self.ftx = ftx
        self.source_prices = []
        self.renko_prices = []
        self.renko_directions = []
        self.renko_prices_for_calculation = []
        self.renko_directions_for_calculation = []
        #self.current_capital = 1000
        self.current_capital = self.get_usd_balance()
        log.info(f'initial balance: {self.current_capital}')
        
        # trend following params
        self.position_data = {"trade_direction": "", "prices_opened": []}
        self.profit = []
        self.capital_history = []
        # self.close_price = pd.DataFrame(read_close_prices_and_times(prices_file))
        self.close_price = pd.DataFrame(get_initial_history(self.market, trailing_history_window))
        self.trailing_history_window = trailing_history_window
        self.min_recalculation_period = min_recalculation_period
        self.last_recalculation_index = 0

        self.candles_since_recalculation = 0

        self.set_brick_size(auto=True)

        prices = self.close_price.iloc[self.trailing_history_window:, 4]

        if len(prices) > 0:
            # Init by start values
            self.source_prices = prices
            self.renko_prices.append(float(prices.iloc[-1]))
            self.renko_directions.append(0)

    def get_usd_balance(self) -> float:
        balances = self.ftx.get_balances()
        for b in balances:
            if b['coin'] == 'USD':
                return b['free']
        return 0

    def calculate_optimal_brick_size(self):
        # self.last_recalculation_index = current_candle
        # Function for optimization
        def evaluate_renko(brick, history, column_name):
            self.set_brick_size(brick_size=brick, auto=False)
            self.build_history_for_calculation(history)
            return self.evaluate_for_calculation(history)[column_name]

        # Get ATR values (it needs to get boundaries)
        # Drop NaNs
        atr = talib.ATR(high=np.double(self.close_price.iloc[-self.trailing_history_window:, 2]),
                        low=np.double(self.close_price.iloc[-self.trailing_history_window:, 3]),
                        close=np.double(self.close_price.iloc[-self.trailing_history_window:, 4]),
                        timeperiod=14)
        atr = atr[np.isnan(atr) == False]

        # Get optimal brick size as maximum of score function by Brent's (or similar) method
        # First and Last ATR values are used as the boundaries
        result = opt.fminbound(lambda x: -evaluate_renko(brick=x,
                                                       history=self.close_price.iloc[-self.trailing_history_window:, 4],
                                                       column_name='score'),
                             np.min(atr),
                             np.max(atr),
                             disp=0)
        log.info(f'calculated optimal brick size: {result}')
        return result

    # Setting brick size. Auto mode is preferred, it uses history
    def set_brick_size(self, HLC_history=None, auto=True, brick_size=10.0):
        if auto == True:
            # test = self.get_close_price().iloc[:, [2, 3, 4]]
            # self.brick_size = self.__get_optimal_brick_size(HLC_history=test)
            self.brick_size = self.calculate_optimal_brick_size()
        else:
            self.brick_size = brick_size
        return self.brick_size

    def __trend_following_strategy(self):
        renko_price = self.renko_prices[-1]
        if self.renko_directions[-1] != 0 and self.renko_directions[-2] == self.renko_directions[-1]:
            # direction matches previous, then open position in following direction
            if not self.position_data["trade_direction"]:
                size = self.current_capital/renko_price
                if self.renko_directions[-1] == 1:
                    position_side = "long"
                    log.info(f'long: price={renko_price} size={size}')
                    # self.ftx.place_order(
                    #         market=self.market,
                    #         side=ftx.buy,
                    #         price=renko_price,
                    #         size=size,
                    #         type=ftx.limit,
                    # )
                else:
                    position_side = "short"
                    log.info(f'short: price={renko_price} size={size}')
                    # self.ftx.place_order(
                    #         market=self.market,
                    #         side=ftx.sell,
                    #         price=renko_price,
                    #         size=size,
                    #         type=ftx.limit,
                    # )
                self.position_data["trade_direction"] = position_side
            self.position_data["prices_opened"].append(renko_price)
        else:
            # position direction has changed, close open order and calculate capital
            log.info('canceling orders, closing positions, price={renko_price}')
            # self.ftx.cancel_orders(market=self.market)
            # self.ftx.close_positions(self.market)

            # profit = 0
            # position_divider = 3
            # for price in self.position_data["prices_opened"][:position_divider]:
            #     if self.position_data["trade_direction"] == 'long':
            #         profit += renko_price/price * \
            #             (self.current_capital/position_divider) - \
            #             self.current_capital/position_divider*1.0002
            #     else:
            #         profit += price/renko_price * \
            #             (self.current_capital/position_divider) - \
            #             self.current_capital/position_divider*1.0002
            
            # self.current_capital += profit
            self.current_capital = self.get_usd_balance()
            log.info(f'updated balance: {self.current_capital}')
            self.capital_history.append(self.current_capital)
            self.position_data["trade_direction"] = None
            self.position_data["prices_opened"] = []
            # recalculate brick size
            #if candle_index - self.last_recalculation_index > self.min_recalculation_period:
            if self.candles_since_recalculation > self.min_recalculation_period:
                self.brick_size = self.calculate_optimal_brick_size()

    def on_new_candle(self, candle):
        # convert ftx candle to binance candle bc close_price is binance format
        self.close_price = self.close_price.append([[
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
        self.__renko_rule(self.close_price.iloc[-1, 4])

    def __renko_rule(self, last_price):
        log.info(f'running renko rule on last price: {last_price}')
        self.candles_since_recalculation += 1
        # Get the gap between two prices
        gap_div = int(
            float(last_price - self.renko_prices[-1]) / self.brick_size)
        is_new_brick = False
        start_brick = 0
        num_new_bars = 0

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
                self.renko_prices.append(
                    self.renko_prices[-1] + 2 * self.brick_size * np.sign(gap_div))
                self.renko_directions.append(np.sign(gap_div))
            # else:
            # num_new_bars = 0

            if is_new_brick:
                # Add each brick
                for _ in range(start_brick, np.abs(gap_div)):
                    brick_price = self.renko_prices[-1] + self.brick_size * np.sign(gap_div)
                    log.info(f'new brick price: {brick_price}')
                    self.renko_prices.append(brick_price)
                    self.renko_directions.append(np.sign(gap_div))
                self.__trend_following_strategy()
            else:
                log.info('no new bricks')

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

            # print(self.renko_prices)

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

    def evaluate(self, method='simple'):
        balance = 0
        sign_changes = 0
        price_ratio = len(self.source_prices) / len(self.renko_prices)

        if method == 'simple':
            for i in range(2, len(self.renko_directions)):
                if self.renko_directions[i] == self.renko_directions[i - 1]:
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
            return {'balance': balance, 'sign_changes:': sign_changes,
                    'price_ratio': price_ratio, 'score': score}

    def get_renko_prices(self):
        return self.renko_prices

    def get_sma(self):
        return self.sma

    def get_balance(self):
        # print(self.close_price)
        return self.current_capital, 1000 - min(self.capital_history), self.brick_size

    def get_renko_directions(self):
        return self.renko_directions

    def plot_renko(self, col_up='g', col_down='r'):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        ax.set_title('Renko chart')
        ax.set_xlabel('Renko bars')
        ax.set_ylabel('Price')

        # Calculate the limits of axes
        ax.set_xlim(0.0,
                    len(self.renko_prices) + 1.0)
        ax.set_ylim(np.min(self.renko_prices) - 3.0 * self.brick_size,
                    np.max(self.renko_prices) + 3.0 * self.brick_size)

        # Plot each renko bar
        for i in range(1, len(self.renko_prices)):
            # Set basic params for patch rectangle
            col = col_up if self.renko_directions[i] == 1 else col_down
            x = i
            y = self.renko_prices[i] - \
                self.brick_size if self.renko_directions[i] == 1 else self.renko_prices[i]
            height = self.brick_size

            # Draw bar with params
            ax.add_patch(
                patches.Rectangle(
                    (x, y),  # (x,y)
                    1.0,  # width
                    self.brick_size,  # height
                    facecolor=col
                )
            )

        plt.show()
