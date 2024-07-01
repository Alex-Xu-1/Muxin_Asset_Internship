import numpy as np
import pandas as pd
import datetime
import os
import matplotlib as mlt
import matplotlib.pyplot as plt
from typing import Any, Tuple

from quantvale import backtest
from quantvale import access
from quantvale.stdsrc import Attr
from quantvale import utils
from quantvale import error
from quantvale import useful
from quantvale import logs
from qvscripts.index import for_indexdb
from qvfactors import rawfactors

for_indexdb.ready()
clt = access.get_default_access_client()

'''
虽然用backtest.BaseBacktestTask,但实际用于回测数据下载
'''
class Task(backtest.BaseBacktestTask):
    def execute(self):
        try:
            exe_start_time = utils.now()
            self.QsCodes, self.args, self.taskID, self.data_dir\
                    = list(itemgetter('QSCode', 'Arguments', 'TASKID', 'DataPath')(self.params))


            self.progress(50)



            self.progress(100)
            self.chrono.stop('End')
            # 报告
            self.end(dict(
                RunTime=dict(
                    start=utils.strdatetime(exe_start_time),
                    stop=utils.now_str(),
                    total=self.chrono.total(),
                    chrono=self.chrono.recall(),
                ),
            ))

        except Exception as e:
            self.log.record(error.get_traceback())
            self.end_with_exception()
