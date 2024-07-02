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

        df_merged['if_ST'].fillna(0, inplace=True)
        df_merged['if_delist_period'].fillna(0, inplace=True)

        self.df_merged = df_merged
        #================================#

    def __del__(self):
        pass

    #================================================#
    def on_func(self,
                date: datetime.date,
                weights: pd.Series) -> pd.Series:

        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
        # TODO: 遴选新成分(权重%)

        mask = ~self.df_merged.index.get_level_values(1).astype(str).str.endswith('BJ')
        
        df_filtered = self.df_merged[mask]
        df_filtered = df_filtered[df_filtered['if_listing'] == 1]
        
        df_filtered = df_filtered[df_filtered['if_delist_period'] == 0]
        df_filtered = df_filtered[df_filtered['close'] >= 1]
        df_min_400 = df_filtered[df_filtered['if_ST'] == 0]

        # Filter out the 400 minimum market value stocks in each day
        df_min_400 = df_filtered.groupby(level=0).apply(lambda x: x.nsmallest(400, 'mkt_val'))

        df_min_400 = df_min_400.reset_index(level=0, drop=True)

        df_min_400 = df_min_400.reset_index()

        df_sec = pd.DataFrame(weights)

        selected_stocks = df_min_400[df_min_400['date'] == date]['qscode']

        self.log.record(selected_stocks)

        new_weights = weights[weights.index.isin(selected_stocks)]
        new_weights[:] = 1/400

        self.log.record(new_weights)

        return new_weights
