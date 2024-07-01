# Testing initial system running fucntionality
# Trying a backtest

from quantvale.bt_pilots.index_alpha.INDEX_ALPHA_BT_V7 import TaskProto

from .strategy import Strategy


class Task(TaskProto):
    def execute(self):
        self.delegate(Strategy)