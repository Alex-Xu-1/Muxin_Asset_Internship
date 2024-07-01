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
                 log: tasksys.TaskLog,
                 params: dict,
                 **kwargs,
                 ):
        self.log = log
        self.params = params
        self.args = self.params['Arguments']
        #
        self.trade_cal: list = kwargs['trade_cal']
        self.all_qvcodes: list = kwargs['all_qvcodes']

        #================================#

        # 预载指标
        self.df_indic_day_bars_of_qvcodes: Dict[str, pd.DataFrame] = {}
        self.dfshm_indic_day_bars_of_qvcodes: Dict[str, utils.DataFrameSharedMemory] = {}
        for qvcode in self.all_qvcodes:
            self.df_indic_day_bars_of_qvcodes[qvcode], self.dfshm_indic_day_bars_of_qvcodes[qvcode] = \
                preload.get_preload_bars_from_shm(
                    preload_id=f"{self.params['STRATEGY']}_INDIC",
                    qscode=qvcode,
                    cycle='day',
                    adjfq=useful.AdjFQ.QFQ)

    def __del__(self):
        for qvcode, dfshm in self.dfshm_indic_day_bars_of_qvcodes.items():
            dfshm.close()

    #================================================#

    def on_func(self,
                date: datetime.date,
                weights: pd.Series) -> pd.Series:
        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
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

        #////////////////////////////////////////////////////////////////////////#

        if len(new_weights) > 0:    # 最小持股数
            new_weights_sum = new_weights.sum()
            if new_weights_sum > 0:
                new_weights = new_weights / new_weights_sum * 100
        else:
            new_weights = pd.Series(name='weight', dtype=float)

        return new_weights
