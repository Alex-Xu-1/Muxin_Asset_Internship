# Copyright (c) 2022 MX
# 邢子文(XING-ZIWEN) <ziwen.xing@muxinasset.com>
# STAMP|380818|380823|380909


import time
from typing import List, Dict

import numpy as np
import pandas as pd

from quantvale import utils
from quantvale import useful
from quantvale import backtest
from quantvale import preload

from .pl_calc import *


'''
PL: 计算数据预载, 可自定义计算各种数据
'''


PRELOADID_STOCK_CODES = '__PL__A__STOCK_CODES'
PRELOADID_FULL_DAY_BARS = '__PL__A__FULL_DAY_BARS'


class Task(backtest.BaseBacktestTask):
    def execute(self):
        try:
            stages = [
                20,    # 票池
                80,    # 个股计算及共享
            ]
            yet_stages = [sum(stages[:i+1]) for i in range(len(stages))]
            progress_appear_num = 10

            self.chrono.stop('Init')
            #========================================#

            if 'UseIndexDB' in self.params:  # 指定范围

                if not self.params['UseIndexDB']:   # 股池
                    df_pool = pd.read_csv(self.params['Pool'], encoding='gbk', index_col='qvcode')
                    all_qvcodes: list = list(df_pool.index)
                    #
                    self.log.record(f'df_pool : {df_pool.shape}')

                else:  # 指数库

                    from qvscripts.index import for_indexdb
                    for_indexdb.ready()

                    # 指数: 成分
                    date_constituents_dict = for_indexdb.fetch_index_constituents_by_period(
                        code=self.params['QSCode'],
                        begin_date=self.begin_date,
                        end_date=self.end_date)

                    self.log.record(f'date_constituents_dict : {len(date_constituents_dict)}')

                    # 所有成份
                    all_constituents_qvcodes_set = set()
                    for date, constituents in date_constituents_dict.items():
                        for constituent in constituents:
                            all_constituents_qvcodes_set.add(constituent['qvcode'])

                    all_qvcodes: list = list(all_constituents_qvcodes_set)
                    all_qvcodes.sort()

            else:   # 全市场
                all_qvcodes = utils.AnyObjectSharedMemory.get(PRELOADID_STOCK_CODES)

            self.log.record(f'all_qvcodes : {len(all_qvcodes)}')

            self.chrono.stop('LoadStockCodes')
            #========================================#
            self.progress(yet_stages[0])  # progress
            #========================================#

            dfshm_indic_day_bars_of_qvcodes: Dict[str, utils.DataFrameSharedMemory] = {}

            finish_count = 0
            for qvcode in all_qvcodes:

                df_full_day_bars, dfshm = preload.get_preload_bars_from_shm(
                    preload_id=PRELOADID_FULL_DAY_BARS,
                    qscode=qvcode,
                    cycle='day',
                    adjfq=useful.AdjFQ.QFQ)
                df_full_day_bars = df_full_day_bars.copy()  # MUST BE `copy`
                dfshm.close()

                #================================#

                df_indic_day_bars = calc_indic(self.log, qvcode, df_full_day_bars)

                dfshm_indic_day_bars_of_qvcodes[qvcode] = preload.set_preload_bars_to_shm(
                    preload_id=f"{self.params['STRATEGY']}_INDIC",
                    qscode=qvcode,
                    cycle='day',
                    adjfq=useful.AdjFQ.QFQ,
                    df_bars=df_indic_day_bars)

                #================================#

                # progress
                finish_count += 1
                if len(all_qvcodes) < progress_appear_num \
                        or finish_count % int(len(all_qvcodes)/progress_appear_num) == 0:
                    self.progress(yet_stages[0] + finish_count/len(all_qvcodes) * stages[1])

            self.chrono.stop('Indic')
            #========================================#
            self.progress(yet_stages[1])  # progress
            #========================================#

            # Wait #
            try:
                while True:
                    time.sleep(5)
            except:
                self.log.record('Terminate', exc_info=True)

            #========================================#

            for qvcode, dfshm in dfshm_indic_day_bars_of_qvcodes.items():
                dfshm.destroy()

            self.chrono.stop('End')
            #========================================#

        except:
            from quantvale import error
            print(error.get_traceback())
            self.end_with_exception()
