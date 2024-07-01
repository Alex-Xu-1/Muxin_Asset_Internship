# Copyright (c) 2022 MX
# 邢子文(XING-ZIWEN) <ziwen.xing@muxinasset.com>
# STAMP|380929


import os
import json


Path = 'builtin_strategies'
STRATEGY = 'TRD_PNL_CTA_TEST'

ParamsFile = 'TRD_PNL_CTA_TEST_BT' + '.json'


if __name__ == "__main__":

    path = os.path.join(Path, STRATEGY)
    path = os.path.join(path, 'ParamsFiles')
    path = os.path.join(path, ParamsFile)

    with open(path, encoding='utf-8') as fp:
        params = json.load(fp)
        print(params)

    from quantvale.builtin_strategies.TRD_PNL_CTA_TEST.BT import Task
    Task(params).execute()
