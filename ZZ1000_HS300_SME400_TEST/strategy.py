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
                 params: dict,
                 **kwargs,
                 ):
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

        mask = ~self.df_merged.index.get_level_values(1).astype(str).str.endswith('BJ')
        df_filtered = self.df_merged[mask]
        df_filtered = df_filtered[df_filtered['if_listing'] == 1]
        df_filtered.drop(columns=['if_listing'], inplace=True)

        df_min_400 = self.df_filtered.groupby(level=0).apply(lambda x: x.nsmallest(400, 'mkt_val'))
        self.df_min_400 = df_min_400
        #================================#

    def __del__(self):
        pass

    #================================================#
    def on_func(self,
                date: datetime.date,
                weights: pd.Series) -> pd.Series:

        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
        # TODO: 遴选新成分(权重%)

        # Filter df_min_400 for the specific date, this returns a new dataframe containing the 400 selected stocks of today
        df_spec = df_min_400.loc[date]
        remove_rows = pd.DataFrame()
        flag = self.trade_cal.index(date)

        for index, row in df_spec.iterrows():
            stkcd = index[1] # iteratively select the stock code from the portfolio
            close_price = row['close']
            if_ST = row['if_ST']
            if_delist = row['if_delist_period']
            if_suspend = row['if_trade_suspend']
            up_limit = row['limit_up']
            down_limit = row['limit_down']
            
            idx_today = self.trade_cal.index(date)
            idx_yesterday = self.trade_cal[idx_today - 1]

            # check if the stock is suspended in that day
            if if_suspend == 1:
                remove_rows
            # check if close_price reaches the limit_up or limit_down
            elif up_limit == close_price or down_limit == close_price:
                remove_rows



        
         
        
        df_filtered = df_filtered[df_filtered['if_delist_period'] != 1]
        df_filtered = df_filtered[df_filtered['close'] >= 1]
        df_filtered = df_filtered[df_filtered['if_ST'] != 1]

        df_min_400 = df_min_400.reset_index(level=0, drop=True)

        df_min_400 = df_min_400.reset_index()

        df_sec = pd.DataFrame(weights)

        selected_stocks = df_min_400[df_min_400['date'] == date]['qscode']

        self.log.record(selected_stocks)

        new_weights = weights[weights.index.isin(selected_stocks)]
        new_weights[:] = 1/400

        self.log.record(new_weights)

        return new_weights
