# Copyright (c) 2022 MX
# 邢子文(XING-ZIWEN) <ziwen.xing@muxinasset.com>
# STAMP|380907


from quantvale.bt_pilots.index_alpha.INDEX_ALPHA_BT_V7 import TaskProto

from .strategy import Strategy


class Task(TaskProto):
    def execute(self):
        self.delegate(Strategy)
