# Copyright (c) 2022 MX
# 邢子文(XING-ZIWEN) <ziwen.xing@muxinasset.com>
# STAMP|381012


import datetime
from typing import List, Dict

import numpy as np
import pandas as pd

from quantvale import utils
from quantvale import tasksys
from quantvale import backtest


class Strategy:
    def __init__(self,
                 params: dict,
                 taskdir: str,
                 log: tasksys.TaskLog,
                 ** kwargs,
                 ):
        self.params = params
        self.taskdir = taskdir
        self.log = log

        self.trader: backtest.BacktestTrader = kwargs['trader']
        self.df_cyc_bars: pd.DataFrame = kwargs['df_cyc_bars']
        self.df_intraday_cyc_bars: pd.DataFrame = kwargs['df_intraday_cyc_bars']

        self.qscode = self.trader.qscode
        self.multiplier = self.trader.multiplier
        self.capital = self.params['Capital']
        self.args = self.params['Arguments']

        #================================================#
        # TODO: Calc

    #====#====#====#====#====#====#====#====#====#====#====#====#
    '''
    CTA_TEST
    根据昨收和今开确定基准价
    今开相对昨收的涨跌幅绝对值超过`BasePrice_Pct`时, 基准价取今开, 否则取昨收
    基准价的正(负)`OpenTrack_Pct`的价位视为多头(空头)开仓轨道
    基准价的负(正)`CloseTrack_Pct`的价位视为多头(空头)平仓轨道
    注意: 无仓位且今日无平仓动作时才可开仓
    '''

    def on_cycle(self, date: datetime.date, bar: pd.Series):
        # print(date, bar)

        self.pre_close = bar['pre_close']
        self.open = bar['open']
        print(date, 'open ====', self.pre_close, self.open)

        self.open_long_line = None      # 开多轨道
        self.open_short_line = None     # 开空轨道
        self.close_long_line = None     # 平多轨道
        self.close_short_line = None    # 平空轨道

        self.open_pos_flag = False
        self.close_pos_flag = False

    def on_quote(self, dt: datetime.datetime, bar: pd.Series):
        price = bar['close']
        # print(dt, price)

        if self.open_long_line is None:  # 日内首次进入的行情视为开盘行情, 进行初始化
            open_pct_chg = (self.open/self.pre_close - 1)*100
            print('open_pct_chg', open_pct_chg)
            base_price = self.open if abs(open_pct_chg) > self.args['BasePrice_Pct'] else self.pre_close
            print(base_price)

            self.open_long_line = base_price * (1 + self.args['OpenTrack_Pct'] / 100)
            self.open_short_line = base_price * (1 - self.args['OpenTrack_Pct'] / 100)
            self.close_long_line = base_price * (1 - self.args['CloseTrack_Pct'] / 100)
            self.close_short_line = base_price * (1 + self.args['CloseTrack_Pct'] / 100)
            print(self.open_long_line, self.open_short_line, self.close_long_line, self.close_short_line)

        max_openable_vol = int(self.capital / price / self.multiplier)
        # print('max_openable_vol', max_openable_vol)

        if self.trader.pos == 0 and not self.open_pos_flag and not self.close_pos_flag:  # 无仓位&无今开&无今平
            if price > self.open_long_line:
                if self.args['Direction'] >= 0:
                    # 开多
                    self.trader.order(dt, price, max_openable_vol)
                    self.open_pos_flag = True
                elif self.args['Direction'] < 0:
                    # 开多1手
                    self.trader.order(dt, price, 1)
                    self.open_pos_flag = True
            elif price < self.open_short_line:
                if self.args['Direction'] <= 0:
                    # 开空
                    self.trader.order(dt, price, -max_openable_vol)
                    self.open_pos_flag = True
                elif self.args['Direction'] > 0:
                    # 开空1手
                    self.trader.order(dt, price, -1)
                    self.open_pos_flag = True

        elif self.trader.pos > 0:    # 有多头仓位
            if price < self.close_long_line:
                # 平多
                self.trader.order_to(dt, price, 0)
                self.close_pos_flag = True

        elif self.trader.pos < 0:    # 有空头仓位
            if price > self.close_short_line:
                # 平空
                self.trader.order_to(dt, price, 0)
                self.close_pos_flag = True

        if dt.time() == utils.totime('15:00:00'):
            if 'Rebalance' in self.args and self.args['Rebalance']:
                if self.args['Direction'] == 0:
                    self.trader.order_to_capital_size(dt, price, self.capital)
                elif self.args['Direction'] > 0:
                    if self.trader.pos > 0:
                        self.trader.order_to_capital_size(dt, price, self.capital)
                elif self.args['Direction'] < 0:
                    if self.trader.pos < 0:
                        self.trader.order_to_capital_size(dt, price, self.capital)
