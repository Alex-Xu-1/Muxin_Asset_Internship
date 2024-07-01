# Copyright (c) 2022 MX
# 邢子文(XING-ZIWEN) <ziwen.xing@muxinasset.com>
# STAMP|380909


import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from quantvale import utils
from quantvale import useful
from quantvale import tasksys
from quantvale import preload


class Strategy:

    def __init__(self,
                 params: dict,
                 taskdir: str,
                 log: tasksys.TaskLog,
                 **kwargs,
                 ):
        self.params = params
        self.taskdir = taskdir
        self.log = log
        self.args = self.params['Arguments']
        #self.args = slef.params['begin dat], 
        self.trade_cal: List[datetime.date] = kwargs['trade_cal']
        self.all_qvcodes: List[str] = kwargs['all_qvcodes']
        #for key, value in kwargs.items():
        #    self.log.record(f"{key}: {value}")
        ################################################
        if self.all_qvcodes.isin(1, 3):
            self.args
        # 计算

        self.df_indic_day_bars_of_qvcodes: Dict[str, pd.DataFrame] = {}
        for qvcode in self.all_qvcodes:
            print(qvcode)

            PL_PREFIX = '__PL__C__'
            PRELOADID_CODES = PL_PREFIX + 'CODES'           # 全部代码
            PRELOADID_BARS = PL_PREFIX + 'DAYBARS'          # 日级别行情
            df_full_day_bars, dfshm = preload.get_preload_bars_from_shm(
                preload_id=PRELOADID_BARS,
                qscode=qvcode,
                cycle='day',
                adjfq=useful.AdjFQ.HFQ)
            df = df_full_day_bars.copy()  # MUST BE `copy`
            dfshm.close()

            ################################################
            import talib

            df['DIFF'], df['DEA'], df['MACDhist'] = \
                talib.MACD(df['close'],
                           fastperiod=12,
                           slowperiod=26,
                           signalperiod=9)

            self.df_indic_day_bars_of_qvcodes[qvcode] = df
            print(self.df_indic_day_bars_of_qvcodes[qvcode])

    def __del__(self):
        pass

    ################################################################################################

    def on_func(self,
                date: datetime.date,
                weights: pd.Series) -> pd.Series:
        ################################################
        # TODO: 遴选新成分(权重%)

        df_sec = pd.DataFrame(weights)
        pre_date = self.trade_cal[self.trade_cal.index(date)-1]

        indic_list = ['DIFF', 'DEA']
        for qvcode in df_sec.index:
            df_days = self.df_indic_day_bars_of_qvcodes[qvcode]
            for indic in indic_list:
                df_sec.at[qvcode, indic] = df_days.at[date, indic]

        stocks = df_sec[
            (df_sec['DIFF'] > df_sec['DEA'])
        ].index

        new_weights = weights[weights.index.isin(stocks)]

        ################################################

        if len(new_weights) >= 1:    # 最小持股数
            if self.args['WeightMethod'] == 'EQUAL':
                new_weights[:] = 1.0 / float(len(new_weights))
            elif self.args['WeightMethod'] == 'FOLLOW':
                new_weights_sum = new_weights.sum()
                if new_weights_sum > 0:
                    new_weights = new_weights / new_weights_sum
            else:
                raise ValueError('Invalid WeightMethod')
        else:
            new_weights = pd.Series(name='weight', dtype=float)

        return new_weights
