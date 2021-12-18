import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import talib
import random

import scipy.optimize as opt
from utils import *
import pandas as pd

class renko:
    def __init__(self, trailing_history, largest_trailing_history, position_divider, atr_multiple, starting_atr_multiple):
        self.source_prices = []
        self.renko_prices = []
        self.renko_directions = []
        self.renko_prices_for_calculation = []
        self.renko_directions_for_calculation = []
        self.brick_sizes = []
        self.current_capital = 1000

        # trend following params
        self.position_data = {"trade_direction": "", "prices_opened": []}
        self.profit = []
        self.capital_history = []
        self.close_price = pd.DataFrame(read_close_prices_and_times())
        self.trailing_history_window = trailing_history #in minutes
        self.min_recalculation_period = 60*12
        self.last_recalculation_index = 0
        self.largest_trailing_history = largest_trailing_history
        self.number_of_candles_calculations = 9
        self.number_of_trades = 0
        self.number_of_attempted_trades = 0
        self.is_multiple_bricks_in_opposite_direction = False
        self.position_divider = position_divider
        self.atr_multiple = atr_multiple
        self.starting_atr_multiple = starting_atr_multiple
        self.positions_closed_by_stoploss = 0
        self.positions_closed_by_bricks = 0

        self.indexes_of_brick_size_retrace = []

        self.indexes_of_upward_retrace = []
        self.indexes_of_downward_retrace = []

        self.indexes_of_drop_below_then_above_first_brick_in_opposite_direction = []
        self.indexes_of_drop_above_then_below_first_brick_in_opposite_direction = []

        self.candle_indexes_of_upward_retrace = []
        self.stopped_bricks = [0]

        macd, signal, hist = talib.MACD(self.close_price.iloc[:, 4],fastperiod=12, slowperiod=26, signalperiod=9)
        self.macd_hist = hist

        self.long_gains = 0
        self.short_gains = 0

        self.stopped_on_edge = 0
        self.stopped_on_candle = 0

        self.closed_on_edge = 0
        self.closed_on_candle = 0

    def get_close_price(self):
        return self.close_price

    def calculate_optimal_brick_size(self, current_candle, is_initial_calculation):
        if (is_initial_calculation):
            self.last_recalculation_index = current_candle - self.largest_trailing_history
        else:
            self.last_recalculation_index = current_candle - self.trailing_history_window
        # Function for optimization
        def evaluate_renko(brick, history, column_name):
            self.set_brick_size(brick_size=brick, auto=False, is_initial_calculation=is_initial_calculation)
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
    def set_brick_size(self, is_initial_calculation, HLC_history=None, auto=True, brick_size=10.0):
        if auto == True:
            # test = self.get_close_price().iloc[:, [2, 3, 4]]
            # self.brick_size = self.__get_optimal_brick_size(HLC_history=test)
            self.brick_size = self.calculate_optimal_brick_size(current_candle=self.largest_trailing_history, is_initial_calculation=True)
            self.brick_sizes.append(self.brick_size)
        else:
            self.brick_size = brick_size
        return self.brick_size

    def __macd_strategy(self, candle_index, last_price):
        profit = 0
        if self.position_data["trade_direction"] == 'long' and self.macd_hist[self.largest_trailing_history + candle_index] < 0 and self.macd_hist[self.largest_trailing_history + candle_index -1] > 0:
            profit += (last_price) / self.position_data["prices_opened"][0] * (self.current_capital) - self.current_capital*1.0002
            if profit != 0:
                self.current_capital += profit
                self.capital_history.append(self.current_capital)
                self.position_data["trade_direction"] = None
                self.position_data["prices_opened"] = []
            self.position_data["trade_direction"] == 'short'
            self.position_data["prices_opened"].append(last_price)
        elif self.position_data["trade_direction"] == 'short' and self.macd_hist[self.largest_trailing_history + candle_index] > 0 and self.macd_hist[self.largest_trailing_history + candle_index -1] < 0:
            profit += (self.position_data["prices_opened"][0]) / last_price * (self.current_capital) - self.current_capital * 1.0002
            if profit != 0:
                self.current_capital += profit
                self.capital_history.append(self.current_capital)
                self.position_data["trade_direction"] = None
                self.position_data["prices_opened"] = []
            self.position_data["trade_direction"] == 'short'
            self.position_data["prices_opened"].append(last_price)

    def __trend_following_strategy(self, candle_index, last_price, atr):
        renko_price = self.renko_prices[-1]
        prev_renko_price = self.renko_prices[-2]
        position_side = None
        if self.renko_directions[-1] != 0 and self.renko_directions[-2] == self.renko_directions[-1] and not self.is_multiple_bricks_in_opposite_direction and self.stopped_bricks[-1] != len(self.renko_prices):
            # direction matches previous, then open position in following direction
            is_missed_trade = random.random() >= 0.75
            self.current_trend_missed=is_missed_trade
            if not self.position_data["trade_direction"]:
                self.number_of_attempted_trades += 1
                if self.renko_directions[-1] == 1:
                    #is next candle low below renko price
                    if(float(self.close_price.iloc[self.largest_trailing_history + candle_index + 1, 3]) <= renko_price):
                        entry_price = renko_price
                    else:
                        entry_price = float(self.close_price.iloc[self.largest_trailing_history + candle_index + 1, 4])
                    position_side = "long"
                    self.position_data["starting_stop_loss"] = renko_price - self.starting_atr_multiple * atr
                else:
                    #is next candle hight aboive renko price
                    if(float(self.close_price.iloc[self.largest_trailing_history + candle_index + 1, 2]) >= renko_price):
                        entry_price = renko_price
                    else:
                        entry_price = float(self.close_price.iloc[self.largest_trailing_history + candle_index + 1, 4])
                    position_side = "short"
                    self.position_data["starting_stop_loss"] = renko_price + self.starting_atr_multiple * atr
                self.position_data["trade_direction"] = position_side
                self.number_of_trades += 1
                self.position_data["prices_opened"].append(entry_price)
        else:
            self.current_trend_missed = False
            if self.position_data["trade_direction"]:
                #position direction has changed, close open order and calculate capital
                profit = 0
                position_divider = self.position_divider

                short_position_close_price = renko_price if float(self.close_price.iloc[self.largest_trailing_history + candle_index + 1 ,2]) >= renko_price else float(self.close_price.iloc[self.largest_trailing_history + candle_index + 1 ,4])
                long_position_close_price = renko_price if float(self.close_price.iloc[self.largest_trailing_history + candle_index + 1 ,3]) <= renko_price else float(self.close_price.iloc[self.largest_trailing_history + candle_index + 1 ,4])

                for price in self.position_data["prices_opened"][:position_divider]:
                    if self.position_data["trade_direction"] == 'long':
                        if long_position_close_price == renko_price:
                            self.closed_on_edge += 1
                        else:
                            self.closed_on_candle += 1
                        profit += long_position_close_price/price * (self.current_capital/position_divider) - self.current_capital/position_divider*1
                        self.long_gains += long_position_close_price/price *(self.current_capital/position_divider) - self.current_capital/position_divider*1
                    else:
                        if short_position_close_price == renko_price:
                            self.closed_on_edge += 1
                        else:
                            self.closed_on_candle += 1
                        profit += price/short_position_close_price * \
                            (self.current_capital/position_divider) - \
                            self.current_capital/position_divider*1
                        self.short_gains += price/short_position_close_price * \
                            (self.current_capital/position_divider) - \
                            self.current_capital/position_divider*1
                self.current_capital += profit
                self.capital_history.append(self.current_capital)
                self.position_data["trade_direction"] = None
                self.position_data["prices_opened"] = []
                self.positions_closed_by_bricks += 1
            # recalculate brick size
            if candle_index - self.last_recalculation_index > self.min_recalculation_period:
                self.brick_size = self.calculate_optimal_brick_size(current_candle=candle_index+self.trailing_history_window, is_initial_calculation=False)
            if self.is_multiple_bricks_in_opposite_direction:
                self.is_multiple_bricks_in_opposite_direction = False
                self.__trend_following_strategy(candle_index, last_price, atr)

    def __renko_rule(self, last_price, candle_index):
        # Get the gap between two prices
        gap_div = int(
            float(last_price - self.renko_prices[-1]) / self.brick_size)
        is_new_brick = False
        start_brick = 0
        num_new_bars = 0
        is_direction_opposite = False

        self.macd_cross_up_down = False
        self.macd_cross_down_up = False

        atr = talib.ATR(high=np.double(self.close_price.iloc[
                                       self.largest_trailing_history + candle_index - 15:self.largest_trailing_history + candle_index,
                                       2]),
                        low=np.double(self.close_price.iloc[self.largest_trailing_history + candle_index - 15:
                                                            self.largest_trailing_history + candle_index, 3]),
                        close=np.double(self.close_price.iloc[self.largest_trailing_history + candle_index - 15:
                                                              self.largest_trailing_history + candle_index, 4]),
                        timeperiod=14)
        atr = atr[np.isnan(atr) == False][0]
        renko_price = self.renko_prices[-1]
        if self.position_data["trade_direction"]:
            profit = 0
            # if self.position_data["trade_direction"] == 'long':
                # self.position_data["stop_loss"] = renko_price - self.brick_size
                # if last_price <= self.position_data["stop_loss"]:
                #     long_position_close_price = self.position_data["stop_loss"] if float(self.close_price.iloc[self.largest_trailing_history + candle_index + 1, 3]) <= self.position_data["stop_loss"] else self.close_price.iloc[self.largest_trailing_history + candle_index + 1, 4]
                #     if long_position_close_price == self.position_data["stop_loss"]:
                #         self.stopped_on_edge+=1
                #     else:
                #         self.stopped_on_candle+=1
                #     for price in self.position_data["prices_opened"][:self.position_divider]:
                #         profit += (long_position_close_price)/price * (self.current_capital/self.position_divider) - self.current_capital/self.position_divider*1.0002
                #     self.current_capital+=profit
                #     self.long_gains+=profit
                #     self.position_data["trade_direction"] = None
                #     self.position_data["prices_opened"] = []
                #     self.capital_history.append(self.current_capital)
                #     self.positions_closed_by_stoploss +=1
                #     self.stopped_bricks.append(len(self.renko_prices))

                # if last_price <= self.position_data['starting_stop_loss']:
                #     long_position_close_price = self.position_data["starting_stop_loss"] if float(self.close_price.iloc[self.largest_trailing_history + candle_index + 1, 3]) <= self.position_data["starting_stop_loss"] else self.close_price.iloc[self.largest_trailing_history + candle_index + 1, 4]
                #     if long_position_close_price == self.position_data["starting_stop_loss"]:
                #         self.stopped_on_edge+=1
                #     else:
                #         self.stopped_on_candle+=1
                #
                #     for price in self.position_data["prices_opened"][:self.position_divider]:
                #         profit += (long_position_close_price) / price * (self.current_capital / self.position_divider) - self.current_capital / self.position_divider * 1
                #     self.current_capital += profit
                #     self.long_gains += profit
                #     self.position_data["trade_direction"] = None
                #     self.position_data["prices_opened"] = []
                #     self.capital_history.append(self.current_capital)
                #     self.positions_closed_by_stoploss += 1
                #     self.stopped_bricks.append(len(self.renko_prices))
            # else:
                # self.position_data["stop_loss"] = renko_price + self.brick_size
                # if last_price >= self.position_data["stop_loss"]:
                #     short_position_close_price = (self.position_data["stop_loss"])  if float(self.close_price.iloc[(self.largest_trailing_history + candle_index + 1), 2]) >= (self.position_data["stop_loss"]) else self.close_price.iloc[self.largest_trailing_history + candle_index + 1, 4]
                #     if short_position_close_price == self.position_data["stop_loss"]:
                #         self.stopped_on_edge+=1
                #     else:
                #         self.stopped_on_candle+=1
                #     for price in self.position_data["prices_opened"][:self.position_divider]:
                #         profit += price/(short_position_close_price) * (self.current_capital/self.position_divider) - self.current_capital/self.position_divider*1.0002
                #     self.current_capital+=profit
                #     self.short_gains+=profit
                #     self.position_data["trade_direction"] = None
                #     self.position_data["prices_opened"] = []
                #     self.capital_history.append(self.current_capital)
                #     self.positions_closed_by_stoploss+=1
                #     self.stopped_bricks.append(len(self.renko_prices))
                # if last_price >= self.position_data['starting_stop_loss']:
                #     short_position_close_price = (self.position_data["starting_stop_loss"])  if float(self.close_price.iloc[(self.largest_trailing_history + candle_index + 1), 2]) >= (self.position_data["starting_stop_loss"]) else self.close_price.iloc[self.largest_trailing_history + candle_index + 1, 4]
                #     if short_position_close_price == self.position_data["starting_stop_loss"]:
                #         self.stopped_on_edge+=1
                #     else:
                #         self.stopped_on_candle+=1
                #     for price in self.position_data["prices_opened"][:self.position_divider]:
                #         profit += price/(short_position_close_price) * (self.current_capital/self.position_divider) - self.current_capital/self.position_divider*1
                #     self.current_capital+=profit
                #     self.short_gains+=profit
                #     self.position_data["trade_direction"] = None
                #     self.position_data["prices_opened"] = []
                #     self.capital_history.append(self.current_capital)
                #     self.positions_closed_by_stoploss+=1
                #     self.stopped_bricks.append(len(self.renko_prices))

        if self.renko_directions[-1] == 1 and self.renko_directions[-2] == -1:
            if last_price < self.renko_prices[-1]:
                self.is_dropped_below_last_renko = True
            if self.is_dropped_below_last_renko and last_price > self.renko_prices[-1]:
                self.indexes_of_drop_below_then_above_first_brick_in_opposite_direction.append(len(self.renko_prices) - 1) if (len(self.renko_prices) - 1) not in self.indexes_of_drop_below_then_above_first_brick_in_opposite_direction else self.indexes_of_drop_below_then_above_first_brick_in_opposite_direction

        if self.renko_directions[-1] == -1 and self.renko_directions[-2] == 1:
            if last_price > self.renko_prices[-1]:
                self.is_dropped_above_last_renko = True
            if self.is_dropped_above_last_renko and last_price < self.renko_prices[-1]:
                self.indexes_of_drop_above_then_below_first_brick_in_opposite_direction.append(len(self.renko_prices) - 1) if (len(self.renko_prices) - 1) not in self.indexes_of_drop_above_then_below_first_brick_in_opposite_direction else self.indexes_of_drop_above_then_below_first_brick_in_opposite_direction
        if self.renko_directions[-1] == 1:
            if last_price < self.renko_prices[-1] - self.brick_size:
                self.indexes_of_upward_retrace.append(len(self.renko_prices) - 1) if (len(self.renko_prices) - 1) not in self.indexes_of_upward_retrace else self.indexes_of_upward_retrace
                self.candle_indexes_of_upward_retrace.append([candle_index + self.largest_trailing_history,self.renko_prices[-1] - self.brick_size]) if [candle_index + self.largest_trailing_history,self.renko_prices[-1] - self.brick_size] not in self.candle_indexes_of_upward_retrace else self.candle_indexes_of_upward_retrace
        else:
            if last_price > self.renko_prices[-1] + self.brick_size:
                self.indexes_of_downward_retrace.append(len(self.renko_prices) - 1) if (len(self.renko_prices) - 1) not in self.indexes_of_downward_retrace else self.indexes_of_downward_retrace

        # if (len(self.renko_prices) - 1) in self.indexes_of_downward_retrace or (len(self.renko_prices) - 1) in self.indexes_of_upward_retrace:
        #     profit = 0
        #     position_divider = self.position_divider
        #     for price in self.position_data["prices_opened"][:position_divider]:
        #         if self.position_data["trade_direction"] == 'long':
        #             profit += (self.renko_prices[-1] - self.brick_size)/price * \
        #                 (self.current_capital/position_divider) - \
        #                 self.current_capital/position_divider*1.0002
        #         else:
        #             profit += price/(self.renko_prices[-1] + self.brick_size) * \
        #                 (self.current_capital/position_divider) - \
        #                 self.current_capital/position_divider*1.0002
        #     self.current_capital += profit
        #     self.capital_history.append(self.current_capital)
        #     self.position_data["trade_direction"] = None
        #     self.position_data["prices_opened"] = []
        profit = 0
        position_divider = self.position_divider
        # for price in self.position_data["prices_opened"][:position_divider]:
        #     if self.position_data["trade_direction"] == 'long' and self.macd_hist[self.largest_trailing_history + candle_index] < 0 and self.macd_hist[self.largest_trailing_history + candle_index -1] > 0:
        #         profit += (last_price)/price * \
        #             (self.current_capital/position_divider) - \
        #             self.current_capital/position_divider*1.0002
        #     elif self.position_data["trade_direction"] == 'short' and self.macd_hist[self.largest_trailing_history + candle_index] > 0 and self.macd_hist[self.largest_trailing_history + candle_index -1] < 0:
        #         profit += price/(last_price) * \
        #             (self.current_capital/position_divider) - \
        #             self.current_capital/position_divider*1.0002
        # if profit != 0:
        #     self.current_capital += profit
        #     self.capital_history.append(self.current_capital)
        #     self.position_data["trade_direction"] = None
        #     self.position_data["prices_opened"] = []
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
                self.brick_sizes.append(self.brick_size)

            # else:
            # num_new_bars = 0

            if is_new_brick:
                for d in range(start_brick, np.abs(gap_div)):
                    self.renko_prices.append(
                        self.renko_prices[-1] + self.brick_size * np.sign(gap_div))
                    self.renko_directions.append(np.sign(gap_div))
                    if (is_direction_opposite):
                        self.is_multiple_bricks_in_opposite_direction = True
                    self.brick_sizes.append(self.brick_size)
                self.__trend_following_strategy(candle_index, last_price, atr)
                self.is_dropped_below_last_renko=False
                self.is_dropped_above_last_renko=False
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

            print('Trades per day:', self.number_of_trades/(len(prices)/60/24), 'position divider', self.position_divider)

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
        trend_continued_after_sign_change = 0

        min_snapshot = 1000
        max_snapshot = 1000
        self.largest_drawdown = 0
        self.drawdawns = []
        for snapshot in self.capital_history:
            if snapshot > max_snapshot:
                max_snapshot = snapshot
                min_snapshot = snapshot
            if snapshot < min_snapshot:
                min_snapshot = snapshot
                largest_drawdown = ((max_snapshot - min_snapshot)/max_snapshot)*100
                if largest_drawdown > self.largest_drawdown:
                    self.largest_drawdown = largest_drawdown
                    self.drawdawns.append(largest_drawdown)

        upward_trend_change_didnt_happen = 0
        downward_trend_change_didnt_happen = 0
        for i in self.indexes_of_upward_retrace:
            if i < len(self.renko_directions) -1 and self.renko_directions[i] == 1:
                if(self.renko_directions[i] == self.renko_directions[i+1]):
                    upward_trend_change_didnt_happen += 1
        for i in self.indexes_of_downward_retrace:
            if i < len(self.renko_directions) -1 and self.renko_directions[i] == -1:
                if(self.renko_directions[i] == self.renko_directions[i+1]):
                    downward_trend_change_didnt_happen += 1
        upward_direction_price_retraced_but_trend_continued_stat = upward_trend_change_didnt_happen/len(self.indexes_of_upward_retrace)
        downward_direction_price_retraced_but_trend_continued_stat = downward_trend_change_didnt_happen/len(self.indexes_of_downward_retrace)

        upward_trend_change_didnt_happen = 0
        for i in self.indexes_of_drop_below_then_above_first_brick_in_opposite_direction:
            if i < len(self.renko_directions) -1 and self.renko_directions[i] == 1:
                if(self.renko_directions[i] == self.renko_directions[i+1]):
                    upward_trend_change_didnt_happen += 1
        uptrend_after_first_brick_price_retraced_but_trend_continued = upward_trend_change_didnt_happen/len(self.indexes_of_drop_below_then_above_first_brick_in_opposite_direction)

        downward_trend_change_didnt_happen = 0
        for i in self.indexes_of_drop_above_then_below_first_brick_in_opposite_direction:
            if i < len(self.renko_directions) -1 and self.renko_directions[i] == -1:
                if(self.renko_directions[i] == self.renko_directions[i+1]):
                    downward_trend_change_didnt_happen += 1
        downtrend_after_first_brick_price_retraced_but_trend_continued = downward_trend_change_didnt_happen/len(self.indexes_of_drop_above_then_below_first_brick_in_opposite_direction)

        for i in range(2, (len(self.renko_directions) - 1)):
            if self.renko_directions[i] == 1 and self.renko_directions[i - 1] == -1 and self.renko_directions[i] == self.renko_directions[i+1]:
                trend_continued_after_sign_change += 1
            if self.renko_directions[i] == -1 and self.renko_directions[i - 1] == 1 and self.renko_directions[i] == self.renko_directions[i+1]:
                trend_continued_after_sign_change += 1


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
            return {'balance': balance,
                    'sign_changes:': sign_changes,
                    'price_ratio': price_ratio,
                    'score': score,
                    'largest_drawdown': self.largest_drawdown,
                    'price dipped one brick size but trend continued up/down': (upward_direction_price_retraced_but_trend_continued_stat, downward_direction_price_retraced_but_trend_continued_stat),
                    'uptrend price dipped down then up after first brick in trend and trend continued': uptrend_after_first_brick_price_retraced_but_trend_continued,
                    'uptrend price dipped up then up down first brick in trend and trend continued': downtrend_after_first_brick_price_retraced_but_trend_continued,
                    'trend_continued after sign change': trend_continued_after_sign_change/sign_changes,
                    'atr multiple': self.atr_multiple,
                    'starting atr multiple': self.starting_atr_multiple,
                    'gains: long/short': (self.long_gains,self.short_gains),
                    'closed by bricks/stoploss': (self.positions_closed_by_bricks,self.positions_closed_by_stoploss),
                    'stopped on edge': self.stopped_on_edge,
                    'stopped on candle': self.stopped_on_candle,
                    'closed on edge': self.closed_on_edge,
                    'closed on candle': self.closed_on_candle,
                    'filled trades percentage': (self.number_of_trades/self.number_of_attempted_trades)
                    }

    def get_renko_prices(self):
        return self.renko_prices

    def get_sma(self):
        return self.sma

    def get_balance(self):
        return self.current_capital, self.trailing_history_window/24/60

    def get_renko_directions(self):
        return self.renko_directions

    def plot_balance(self):
        plt.plot(self.capital_history)
        plt.ylabel('balance')
        plt.show()

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
            height = self.brick_sizes[i]

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
