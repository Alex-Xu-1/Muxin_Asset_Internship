# Copyright (c) 2022 MX
# 邢子文(XING-ZIWEN) <ziwen.xing@muxinasset.com>
# STAMP|381011


from quantvale.bt_pilots.trade_pnl.TRADE_PNL_SINGLE_BT_V4 import TaskProto

from .strategy import Strategy


class Task(TaskProto):
    def execute(self):
        self.delegate(Strategy)
