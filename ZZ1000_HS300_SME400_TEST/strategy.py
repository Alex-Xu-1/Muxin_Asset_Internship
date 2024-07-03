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
        
        beg = pd.to_datetime(self.start_day)
        end = pd.to_datetime(self.end_day)

        #================================#
        # Getting raw data of all SH+SZ A share stocks, including: \
            # if_Listing, daily_Closing_Price, if_ST, is_in_Delisting_Period, \
            # if_Trade_Suspension, Market_Value

        df_listing = rawfactors.get_rawfactor_table_by_date(
                                        db="TEST",
                                        name="xdf_if_listing_qvcode",
                                        begin_date=beg,
                                        end_date=end)

        df_close = rawfactors.get_rawfactor_table_by_date(
                                        db="TEST",
                                        name="xdf_close",
                                        begin_date=beg,
                                        end_date=end)

        df_ST = rawfactors.get_rawfactor_table_by_date(
                                        db="TEST",
                                        name="xdf_if_ST",
                                        begin_date=beg,
                                        end_date=end)

        df_mkt_val = rawfactors.get_rawfactor_table_by_date(
                                        db="TEST",
                                        name="xdf_market_value",
                                        begin_date=beg,
                                        end_date=end)

        df_delist_period = rawfactors.get_rawfactor_table_by_date(
                                        db="TEST",
                                        name="xdf_if_delisting_period",
                                        begin_date=beg,
                                        end_date=end)

        df_if_trade_suspend = rawfactors.get_rawfactor_table_by_date(
                                        db="TEST",
                                        name="xdf_trade_suspension",
                                        begin_date=beg,
                                        end_date=end)

        df_limit_up = rawfactors.get_rawfactor_table_by_date(
                                        db="TEST",
                                        name="xdf_limit_up_price",
                                        begin_date=beg,
                                        end_date=end)

        df_limit_down = rawfactors.get_rawfactor_table_by_date(
                                        db="TEST",
                                        name="xdf_limit_down_price",
                                        begin_date=beg,
                                        end_date=end)

                
        df_listing = df_listing.stack().rename('if_listing')
        df_close = df_close.stack().rename('close')
        df_ST = df_ST.stack().rename('if_ST')
        df_mkt_val = df_mkt_val.stack().rename('mkt_val')
        df_delist_period = df_delist_period.stack().rename('if_delist_period')
        df_if_trade_suspend = df_if_trade_suspend.stack().rename('if_trade_suspend')
        df_limit_up = df_limit_up.stack().rename('limit_up')
        df_limit_down = df_limit_down.stack().rename('limit_down')

        # Merging
        df_merged = pd.concat([df_listing, df_close, df_ST, df_mkt_val, df_delist_period, \
                                df_if_trade_suspend, df_limit_up, df_limit_down], axis=1)

        # mask = ~df_merged.index.get_level_values(1).astype(str).str.endswith('BJ')
        df_filtered = df_merged[~df_merged.index.get_level_values(1).astype(str).str.endswith('BJ')]
        df_min_400 = df_filtered[df_filtered['if_listing'] == 1]
        df_min_400.drop(columns=['if_listing'], inplace=True)

        df_min_400 = df_min_400.groupby(level=0).apply(lambda x: x.nsmallest(400, 'mkt_val'))
        self.df_min_400 = df_min_400
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
        # TODO: 遴选新成分(权重%)

        df_spec = self.df_min_400.loc[date]
        selected_stocks = df_spec.index.get_level_values(1)

        new_weights = pd.Series(0, index=weights.index)

        new_weights[selected_stocks] = 1/400

        # try:
        #     for i in selected_stocks:
        #         new_weights[i] = 1/400
        # except:
        #     self.log.record(i)
                                

        # iteratively go through each stocks in the df_spec, performing weights adjustment based on defined criteria
        for index, row in df_spec.iterrows():

            stkcd = index[1] # iteratively select the stock code from the portfolio
            close_price = row['close']
            if_ST = row['if_ST']
            if_delist = row['if_delist_period']
            if_suspend = row['if_trade_suspend']
            up_limit = row['limit_up']
            down_limit = row['limit_down']
            
            # idx_today = self.trade_cal.index(date)
            # idx_yesterday = self.trade_cal[idx_today - 1]

            # check if the stock is suspended in that day
            if if_suspend == 1:
                # today this stock weight can't be adjusted, we can't trade on this stock since it is suspended. \
                # so this stock remains in the selected 400 stocks and we assign the weight of this stock to be the same as yesterday
                new_weights[stkcd] = weights.get(stkcd)
                continue

            elif up_limit == close_price:
                # we can only sell this stock rather than buy it, so we can only adjust the weight of this stock to be smaller than yesterday
                if if_ST == 1 or close_price < 1 or if_delist == 1:
                    # assign the weight of this stock to be 0
                    new_weights[stkcd] = 0
                    continue
                else:
                    # assign this stock yesterday weight
                    new_weights[stkcd] = weights.get(stkcd)
                    continue

            elif down_limit == close_price:
                # we can only buy this stock rather than sell it, so we can only adjust the weight of this stock to be larger than yesterday
                # so we just simply set it to be yesterday's weight since we cannot sell
                new_weights[stkcd] = weights.get(stkcd)
                continue

            else: # continue normal selection
                if if_ST == 1 or close_price < 1 or if_delist == 1:
                    # assign the weight of this stock to be 0
                    new_weights[stkcd] = 0
                    continue
                else:
                    # assign the weight of this stock to be the same as yesterday
                    new_weights[stkcd] = weights.get(stkcd)
                    continue

        self.log.record(new_weights)

        return new_weights
