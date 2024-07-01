# Copyright (c) 2022 MX
# 邢子文(XING-ZIWEN) <ziwen.xing@muxinasset.com>
# STAMP|380909


import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from quantvale import utils
from quantvale import tasksys

import talib


def calc_indic(log: tasksys.TaskLog,
               qvcode: str,
               df_day_bars: pd.DataFrame) -> pd.DataFrame:
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
    # TODO: 计算个股指标

    try:
        df_day_bars['DIFF'], df_day_bars['DEA'], df_day_bars['MACDhist'] = \
            talib.MACD(df_day_bars['close'],
                       fastperiod=12,
                       slowperiod=26,
                       signalperiod=9)
    except:
        log.record(f'except: {qvcode}', exc_info=True)

    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
    return df_day_bars
