import datetime
import os

from typing import Dict, List

import numpy as np
import pandas as pd

from quantvale import utils
from quantvale import useful
from quantvale import tasksys
from quantvale import preload
from quantvale import access
from quantvale.stdsrc import Attr
from quantvale import error
from quantvale import logs

from qvscripts.index import for_indexdb

from qvfactors import rawfactors

from typing import Any, Tuple

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class Strategy:

    def __init__(self,
                 log: tasksys.TaskLog,
                 taskdir: str,
                 params: dict,
                 weights: pd.Series,
                 **kwargs,
                 ):
        self.taskdir = taskdir
        self.log = log
        self.params = params
        self.start_day = self.params["BeginDate"]
        self.end_day = self.params["EndDate"]
        self.args = self.params['Arguments']
        self.trade_cal: list = kwargs['trade_cal']
        self.all_qvcodes: list = kwargs['all_qvcodes']
        
        beg = pd.to_datetime(self.start_day).date()
        end = pd.to_datetime(self.end_day).date()

        #================================#
        # Getting raw data of all SH+SZ A share stocks, including: \
            # if_Listing, daily_Closing_Price, if_ST, is_in_Delisting_Period, \
            # if_Trade_Suspension, Market_Value

        def get_raw(factor_name, col_rename):
            df = rawfactors.get_rawfactor_table_by_date(
                                        db="TEST",
                                        name=factor_name,
                                        begin_date=beg,
                                        end_date=end)
            return df.stack().rename(col_rename)

        df_listing = get_raw("xdf_if_listing_qvcode", 'if_listing')
        df_close = get_raw("xdf_close", 'close')
        df_ST = get_raw("xdf_if_ST", 'if_ST')
        df_mkt_val = get_raw("xdf_market_value", 'mkt_val')
        df_delist_period = get_raw("xdf_if_delisting_period", 'if_delist_period')
        df_if_trade_suspend = get_raw("xdf_trade_suspension", 'if_trade_suspend')
        df_limit_up = get_raw("xdf_limit_up_price", 'limit_up')
        df_limit_down = get_raw("xdf_limit_down_price", 'limit_down')
        df_hs300_weights = get_raw("xdf_hs300_weight", 'hs300_weights')
        df_zz1000_weights = get_raw("xdf_zz1000_weight", 'zz1000_weights')


        # Merging
        df_merged = pd.concat([df_listing, df_close, df_ST, df_mkt_val, df_delist_period, \
                                df_if_trade_suspend, df_limit_up, df_limit_down], axis=1)

        # mask = ~df_merged.index.get_level_values(1).astype(str).str.endswith('BJ')
        df_filtered = df_merged[~df_merged.index.get_level_values(1).astype(str).str.endswith('BJ')]
        df_filtered = df_filtered[df_filtered['if_listing'] == 1]

        df_min_400 = df_filtered[~(df_filtered['if_ST'] == 1)]
        df_min_400 = df_min_400[df_min_400['close'] >= 1]
        df_min_400 = df_min_400[~(df_min_400['if_delist_period'] == 1)]
        df_min_400.drop(columns=['if_listing'], inplace=True)
        df_min_400 = df_min_400.groupby(level=0).apply(lambda x: x.nsmallest(400, 'mkt_val'))

        # create a df that stores only 'close', 'if_trade_suspend', 'limit_up', and 'limit_down'
        df_sel_info = df_filtered.drop(columns=['if_listing', 'if_ST', 'mkt_val', 'if_delist_period'])

        self.old_weights = weights
        self.old_weights[:] = 0
        self.df_sel_info = df_sel_info
        self.df_min_400 = df_min_400
        self.yesterday_stocks = set()

        #================================#

    def __del__(self):
        # self.df_out.index.name = 'date'
        # self.df_out.to_csv(os.path.join(self.taskdir, f"data/size_data.csv"), encoding='gbk')
        pass

    #================================================#
    def on_func(self,
                date: datetime.date,
                weights: pd.Series) -> pd.Series:

        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
        # Get the set of today's stocks
        today_stocks = set(self.df_min_400.loc[date].index.get_level_values(1))

        # Compare with yesterday's stocks and get the three partition
        remain_stocks = self.yesterday_stocks.intersection(today_stocks)
        new_stocks = today_stocks - self.yesterday_stocks
        delete_stocks = self.yesterday_stocks - today_stocks
        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#

        new_weights = pd.Series(0, index=self.old_weights.index)
        ls_today_stocks = list(today_stocks)
        new_weights[ls_today_stocks] = 1

        today_sel_info = self.df_sel_info.loc[date]

        # iteratively go through each stocks in the df_spec, performing weights adjustment based on defined criteria
        for index, row in today_sel_info.iterrows():

            stkcd = index[1] # iteratively select the stock code from the portfolio
            close_price = row['close']
            if_suspend = row['if_trade_suspend']
            up_limit = row['limit_up']
            down_limit = row['limit_down']

            # check if the stock is suspended in that day
            if if_suspend == 1:
                # today this stock weight can't be adjusted, we can't trade on this stock since it is suspended. \
                # so this stock remains in the selected stocks and assign the weight of this stock to be the same as yesterday
                new_weights[stkcd] = self.old_weights.get(stkcd)
                continue

            elif up_limit == close_price:
                # we can only sell this stock rather than buy it, so we can only adjust the weight of this stock to be smaller than yesterday
                if stkcd in delete_stocks:
                    # assign the weight of this delete_stock to be 0
                    new_weights[stkcd] = 0
                else:
                    # assign this stock yesterday weight
                    new_weights[stkcd] = self.old_weights.get(stkcd)
                continue

            elif down_limit == close_price:
                # we can only buy this stock rather than sell it, so we can only adjust the weight of this stock to be larger than yesterday
                if stkcd in new_stocks:
                    new_weights[stkcd] = 1
                else:
                    new_weights[stkcd] = self.old_weights.get(stkcd)
                continue

            else: # continue normal selection
                if stkcd in new_stocks:
                    new_weights[stkcd] = 1
                elif stkcd in delete_stocks:
                    new_weights[stkcd] = 0
                else:
                    new_weights[stkcd] = self.old_weights.get(stkcd)

        # After adjustment, extract the actual stocks today
        actual_stocks_today = set(new_weights[new_weights == 1].index.tolist())

        # Assign the weight of the selected stocks, using equal method
        actual_stocks_num = len(actual_stocks_today)
        if actual_stocks_num > 0:
            weight_per_stock = 1 / actual_stocks_num
            new_weights[list(actual_stocks_today)] = weight_per_stock

        # Update today's final weights to be the old_weights to be used tomorrow
        self.old_weights = new_weights

        # Update yesterday_stocks for the next day
        self.yesterday_stocks = actual_stocks_today

        return new_weights
