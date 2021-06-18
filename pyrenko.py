import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import talib

import scipy.optimize as opt
from utils import *
import pandas as pd


class renko:
    def __init__(self, trailing_history, largest_trailing_history):
        self.source_prices = []
        self.renko_prices = []
        self.renko_directions = []
        self.renko_prices_for_calculation = []
        self.renko_directions_for_calculation = []
        self.current_capital = 1000

        # trend following params
        self.position_data = {"trade_direction": "", "prices_opened": []}
        self.profit = []
        self.capital_history = []
        self.close_price = pd.DataFrame(read_close_prices_and_times())
        self.trailing_history_window = trailing_history #in minutes
        self.min_recalculation_period = 1440
        self.last_recalculation_index = 0
        self.largest_trailing_history = largest_trailing_history
        self.number_of_candles_calculations = 9
        self.number_of_trades = 0
        self.is_multiple_bricks_in_opposite_direction = False

    def get_close_price(self):
        return self.close_price

    def calculate_optimal_brick_size(self, current_candle):
        self.last_recalculation_index = current_candle
        # Function for optimization
        def evaluate_renko(brick, history, column_name):
            self.set_brick_size(brick_size=brick, auto=False)
            self.build_history_for_calculation(history)
            return self.evaluate_for_calculation(history)[column_name]

        close_prices = self.get_close_price()

        # Get ATR values (it needs to get boundaries)
        # Drop NaNs
        atr = talib.ATR(high=np.double(close_prices.iloc[(current_candle - self.trailing_history_window):current_candle, 2]),
                        low=np.double(close_prices.iloc[(current_candle - self.trailing_history_window):current_candle, 3]),
                        close=np.double(close_prices.iloc[(current_candle - self.trailing_history_window):current_candle, 4]),
                        timeperiod=14)
        atr = atr[np.isnan(atr) == False]

        # Get optimal brick size as maximum of score function by Brent's (or similar) method
        # First and Last ATR values are used as the boundaries

        renko_values=[]
        number_of_candles_back=0
        while number_of_candles_back < self.number_of_candles_calculations:
            renko_values.append(opt.fminbound(lambda x: -evaluate_renko(brick=x,
                                                       history=close_prices.iloc[(current_candle - self.trailing_history_window):current_candle - number_of_candles_back, 4],
                                                       column_name='score'),
                             np.min(atr),
                             np.max(atr),
                             disp=0))
            number_of_candles_back += 1
        return np.median(renko_values)

    # Setting brick size. Auto mode is preferred, it uses history
    def set_brick_size(self, HLC_history=None, auto=True, brick_size=10.0):
        if auto == True:
            # test = self.get_close_price().iloc[:, [2, 3, 4]]
            # self.brick_size = self.__get_optimal_brick_size(HLC_history=test)
            self.brick_size = self.calculate_optimal_brick_size(current_candle=self.trailing_history_window)
        else:
            self.brick_size = brick_size
        return self.brick_size

    def __trend_following_strategy(self, candle_index):
        renko_price = self.renko_prices[-1]
        prev_renko_price = self.renko_prices[-2]
        if self.renko_directions[-1] != 0 and self.renko_directions[-2] == self.renko_directions[-1] and not self.is_multiple_bricks_in_opposite_direction:
            # direction matches previous, then open position in following direction
            if not self.position_data["trade_direction"]:
                if self.renko_directions[-1] == 1:
                    position_side = "long"
                else:
                    position_side = "short"
                self.position_data["trade_direction"] = position_side
                self.number_of_trades += 1
            self.position_data["prices_opened"].append(renko_price)
        else:
            # position direction has changed, close open order and calculate capital
            profit = 0
            position_divider = 1
            for price in self.position_data["prices_opened"][:position_divider]:
                if self.position_data["trade_direction"] == 'long':
                    profit += renko_price/price * \
                        (self.current_capital/position_divider) - \
                        self.current_capital/position_divider*1.0002
                else:
                    profit += price/renko_price * \
                        (self.current_capital/position_divider) - \
                        self.current_capital/position_divider*1.0002
            self.current_capital += profit
            self.capital_history.append(self.current_capital)
            self.position_data["trade_direction"] = None
            self.position_data["prices_opened"] = []

            # recalculate brick size
            if candle_index - self.last_recalculation_index > self.min_recalculation_period:
                self.brick_size = self.calculate_optimal_brick_size(current_candle=self.trailing_history_window+candle_index)
                print(self.brick_size)
            self.is_multiple_bricks_in_opposite_direction = False

    def __renko_rule(self, last_price, candle_index):
        # Get the gap between two prices
        gap_div = int(
            float(last_price - self.renko_prices[-1]) / self.brick_size)
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
                self.renko_prices.append(
                    self.renko_prices[-1] + 2 * self.brick_size * np.sign(gap_div))
                self.renko_directions.append(np.sign(gap_div))
            # else:
            # num_new_bars = 0

            if is_new_brick:                # Add each brick
                for d in range(start_brick, np.abs(gap_div)):
                    self.renko_prices.append(
                        self.renko_prices[-1] + self.brick_size * np.sign(gap_div))
                    self.renko_directions.append(np.sign(gap_div))
                    if (is_direction_opposite):
                        self.is_multiple_bricks_in_opposite_direction = True
                self.__trend_following_strategy(candle_index)
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

    # Getting renko on history
    def build_history(self):
        close_prices = self.get_close_price()
        prices = close_prices.iloc[self.largest_trailing_history:, 4]
        print(len(prices)/24/60)

        if len(prices) > 0:
            # Init by start values
            self.source_prices = prices
            self.renko_prices.append(float(prices.iloc[0]))
            self.renko_directions.append(0)

            # For each price in history
            for index, p in enumerate(self.source_prices[1:]):
                self.__renko_rule(float(p), index)

            print('Trades per day:', self.number_of_trades/(len(prices)/60/24))

        return len(self.renko_prices)

    # Getting next renko value for last price
    def do_next(self, last_price):
        if len(self.renko_prices) == 0:
            self.source_prices.append(last_price)
            self.renko_prices.append(last_price)
            self.renko_directions.append(0)
            return 1
        else:
            self.source_prices.append(last_price)
            return self.__renko_rule(last_price)

    # Simple method to get optimal brick size based on ATR
    def __get_optimal_brick_size(self, HLC_history, atr_timeperiod=14):
        brick_size = 0.0

        # If we have enough of data
        if HLC_history.shape[0] > atr_timeperiod:
            brick_size = np.median(talib.ATR(high=np.double(HLC_history.iloc[:, 0]),
                                             low=np.double(
                                                 HLC_history.iloc[:, 1]),
                                             close=np.double(
                                                 HLC_history.iloc[:, 2]),
                                             timeperiod=atr_timeperiod)[atr_timeperiod:])
            self.atr = talib.ATR(high=np.double(HLC_history.iloc[:, 0]), low=np.double(
                HLC_history.iloc[:, 1]), close=np.double(HLC_history.iloc[:, 2]), timeperiod=atr_timeperiod)[atr_timeperiod:]

        return brick_size

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
        return self.current_capital, self.trailing_history_window/24/60

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
