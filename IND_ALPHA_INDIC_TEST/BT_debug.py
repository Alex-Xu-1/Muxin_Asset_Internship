# Copyright (c) 2022 MX
# 邢子文(XING-ZIWEN) <ziwen.xing@muxinasset.com>
# STAMP|380826


import os
import json


Path = 'builtin_strategies'
STRATEGY = 'IND_ALPHA_INDIC_TEST'

ParamsFile = 'POOL_600519_BT' + '.json'
# ParamsFile = 'POOL_3_BT' + '.json'


if __name__ == "__main__":

    path = os.path.join(Path, STRATEGY)
    path = os.path.join(path, 'ParamsFiles')
    path = os.path.join(path, ParamsFile)

    with open(path, encoding='utf-8') as fp:
        params = json.load(fp)
        print(params)

    from quantvale.builtin_strategies.IND_ALPHA_INDIC_TEST.BT import Task
    Task(params).execute()
