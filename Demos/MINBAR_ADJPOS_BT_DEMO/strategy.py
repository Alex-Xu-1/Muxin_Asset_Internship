# Copyright (c) 2024 MX
# 邢子文(XING-ZIWEN) <ziwen.xing@muxinasset.com>
# STAMP|240619


import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from quantvale import utils
from quantvale import useful
from quantvale import tasksys


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
        self.args = self.params['Arguments']
        #
        self.trade_cal: list = kwargs['trade_cal']
        self.all_qvcodes: list = kwargs['all_qvcodes']

    def __del__(self):
        pass

    ################################################################################################

    def before_open(self,
                    date: datetime.date):
        # 开盘前处理
        self.log.record(f'{date} before_open')

    def at_datetime(self,
                    dt: datetime.datetime,
                    weights: pd.Series,
                    ) -> Tuple[bool, pd.Series]:
        ################################################
        # TODO 返回:是否交易,以及调仓权重(长度须和weights保持一致)

        if dt == utils.todatetime('2024-5-15 10:15:00'):
            self.log.record(f'{dt} trade')
            weights[-50:] = 0
            new_weights = weights
            return True, new_weights
        else:
            return False, weights

    def after_close(self,
                    date: datetime.date):
        # 收盘后处理
        self.log.record(f'{date} after_close')
