# Copyright (c) 2024 MX
# 邢子文(XING-ZIWEN) <ziwen.xing@muxinasset.com>
# STAMP|240607


from quantvale.bt_pilots.minbar_adjpos.MINBAR_ADJPOS_BT_V6 import TaskProto

from .strategy import Strategy


class Task(TaskProto):
    def execute(self):
        self.delegate(Strategy)
