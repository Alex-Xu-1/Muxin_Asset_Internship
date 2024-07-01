import datetime

import pandas as pd
import numpy as np

from quantvale import utils
from quantvale import error
from quantvale import useful
from quantvale import logs
from quantvale import access
from quantvale import access_app

clt = access.get_default_access_client()

# Getting trade_day info
trade_cal = clt.trade_cal(utils.date_delta(utils.today(), -30), utils.today())
trade_cal

# Getting stock codes
#stkcd = clt.stock_board_qvcodes_by_date(board='沪深A股', date=utils.today()

begin_date=utils.date_delta(utils.today(), -30)
end_date=utils.today()

# Stock attribute
df = clt.batch_attributes(
    qscodes=['600000.SH', '600004.SH', '600006.SH'],
    begin_date=utils.date_delta(utils.today(), -30),
    end_date=utils.today(),
    attrs=['公司简称', '是否ST', ]
)
df.set_index('qvcode', inplace=True)
df.head()

df_portfolio = access_app.portfolio_equal_weighted_with_access(
    clt,
    qvcodes=['600000.SH', '600004.SH', '600006.SH'],
    begin_date=utils.date_delta(utils.today(), -30),
    end_date=utils.today(),
    exclude_listing_days=5,
    basevalue=1)
df_portfolio