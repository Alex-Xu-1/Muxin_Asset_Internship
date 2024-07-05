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

        self.yesterday_stocks = set()
        self.old_weights = pd.Series()
        
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
        df_sw_industry = get_raw("xdf_sw_industry_level1", 'industry')

        # Merging
        df_merged = pd.concat([df_listing, df_close, df_ST, df_mkt_val, df_delist_period, \
                                df_if_trade_suspend, df_limit_up, df_limit_down], axis=1)

        # mask = ~df_merged.index.get_level_values(1).astype(str).str.endswith('BJ')
        df_filtered = df_merged[~df_merged.index.get_level_values(1).astype(str).str.endswith('BJ')]
        df_filtered = df_filtered[~df_filtered.index.get_level_values(1).astype(str).str.startswith('BJ')]
        df_filtered = df_filtered[df_filtered['if_listing'] == 1]

        df_adjust_info = df_filtered.drop(columns=['if_listing', 'if_ST', 'mkt_val', 'if_delist_period'])

        def theory_min400_select(df):
            df = df[~(df['if_ST'] == 1)]
            df = df[df['close'] >= 1]
            df = df[~(df['if_delist_period'] == 1)]
            df.drop(columns=['if_listing'], inplace=True)
            df = df.groupby(level=0).apply(lambda x: x.nsmallest(400, 'mkt_val'))
            df = df.reset_index(level=0, drop=True)
            return df

        df_theory_min_400 = theory_min400_select(df_filtered)

        # create a df that stores only 'close', 'if_trade_suspend', 'limit_up', and 'limit_down'
        

        
        self.df_adjust_info = df_adjust_info
        self.df_theory_min_400 = df_theory_min_400
        


        #================================#

    def __del__(self):
        # self.df_out.index.name = 'date'
        # self.df_out.to_csv(os.path.join(self.taskdir, f"data/size_data.csv"), encoding='gbk')
        pass

    #================================================#
    def on_func(self,
                date: datetime.date,
                weights: pd.Series) -> pd.Series:

        def suspend_limit_adjust(df, stkcd_list, old_weights, theory_weights, final_weights):
            for i in stkcd_list:

                if df.loc[i, 'if_suspend'] == 1:
                    final_weights[i] = old_weights[i]

                elif df.loc[i, 'limit_up'] == df.loc[i, 'close']:
                    if old_weights[i] >= theory_weights[i]:
                        final_weights[i] = theory_weights[i]
                    elif old_weights[i] < theory_weights[i]:
                        final_weights[i] = old_weights[i]

                elif df.loc[i, 'limit_down'] == df.loc[i, 'close']:
                    if old_weights[i] <= theory_weights[i]:
                        final_weights[i] = theory_weights[i]
                    elif old_weights[i] > theory_weights[i]:
                        final_weights[i] = old_weights[i]
                
                else:
                     
            return final_weights
        

        # Get the set of today's stocks
        theory_min400_stocks_today = list(self.df_theory_min_400.loc[date].index.get_level_values(0))

        # Compare with yesterday's stocks and get the three partition
        # remain_stocks = self.yesterday_stocks.intersection(today_stocks)
        # new_stocks = today_stocks - self.yesterday_stocks
        # delete_stocks = self.yesterday_stocks - today_stocks

        today_adjust_info = self.df_adjust_info.loc[date]
        theory_weights = pd.Series(0, index=weights.index)
        final_weights = pd.Series(0, index=weights.index)
        final_weights[theory_min400_stocks_today] = 1
        
        # Initialize the old_weights in the first day
        if date == self.start_day:
            self.old_weights = weights
            self.old_weights[:] = 0

        suspend_limit_adjust(today_adjust_info, theory_min400_stocks_today, self.old_weights, theory_weights, final_weights)
        
        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
        # Compare with yesterday's stocks and get the three partition
        # remain_stocks = self.yesterday_stocks.intersection(today_stocks)
        # new_stocks = today_stocks - self.yesterday_stocks
        # delete_stocks = self.yesterday_stocks - today_stocks
        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#

        # After adjustment, extract the actual stocks today
        actual_stocks_today = set(final_weights[final_weights == 1].index.tolist())

        # Assign the weight of the selected stocks, using equal method
        actual_stocks_num = len(actual_stocks_today)
        if actual_stocks_num > 0:
            weight_per_stock = 1 / actual_stocks_num
            final_weights[list(actual_stocks_today)] = weight_per_stock

        # Update today's final weights to be the old_weights to be used tomorrow
        self.old_weights = final_weights

        # Update yesterday_stocks for the next day
        self.yesterday_stocks = actual_stocks_today

        self.log.record(print(len(list(actual_stocks_today))))

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Need to add a line that extract all the stocks with non-zero \
            # weights, and then return this small df to the system
        return # ！！！
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!