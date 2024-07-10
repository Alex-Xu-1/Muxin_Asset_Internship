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

        def get_raw(factor_name, col_rename, beg=beg, end=end):
            df = rawfactors.get_rawfactor_table_by_date(
                                        db="TEST",
                                        name=factor_name,
                                        begin_date=beg,
                                        end_date=end)
            return df.stack().rename(col_rename)

        def theory_min400_select(df):
            df = df[~(df['if_ST'] == 1)]
            df = df[df['close'] >= 1]
            df = df[~(df['if_delist_period'] == 1)]
            df.drop(columns=['if_listing'], inplace=True)
            df = df.groupby(level=0).apply(lambda x: x.nsmallest(400, 'mkt_val'))
            df = df.reset_index(level=0, drop=True)
            df['min400_weights'] = 1 / 400
            return df


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
                                df_if_trade_suspend, df_limit_up, df_limit_down, df_sw_industry], axis=1)
        df_zz1000 = pd.concat([df_zz1000_weights, df_sw_industry, df_mkt_val], axis=1)
        df_hs300 = pd.concat([df_hs300_weights, df_sw_industry, df_mkt_val], axis=1)

        all_hs300_stocks =  df_hs300_weights.index.get_level_values(1).unique()

        # mask = ~df_merged.index.get_level_values(1).astype(str).str.endswith('BJ')
        df_filtered = df_merged[~df_merged.index.get_level_values(1).astype(str).str.endswith('BJ')]
        df_filtered = df_filtered[~df_filtered.index.get_level_values(1).astype(str).str.startswith('BJ')]
        df_filtered = df_filtered[df_filtered['if_listing'] == 1]

        df_theory_min400 = theory_min400_select(df_filtered)

        # create a df that stores only 'close', 'mkt_val', 'if_trade_suspend', 'limit_up', and 'limit_down'
        df_adjust_info = df_filtered.drop(columns=['if_listing', 'if_ST', 'if_delist_period'])

        self.df_adjust_info = df_adjust_info
        self.df_theory_min400 = df_theory_min400
        self.all_hs300_stocks = all_hs300_stocks
        self.df_zz1000 = df_zz1000
        self.df_hs300 = df_hs300
        self.df_hs300_weights = df_hs300_weights
        #================================#

    def __del__(self):
        # self.df_out.index.name = 'date'
        # self.df_out.to_csv(os.path.join(self.taskdir, f"data/size_data.csv"), encoding='gbk')
        pass

    #================================================#
    def on_func(self,
                date: datetime.date,
                weights: pd.Series) -> pd.Series:

        def calculate_industry_weight_sum(df, weight_name, date):
            df_temp = df.loc[date].copy()  
            df_today = df_temp.loc[df_temp[weight_name] > 0].copy()  
            # Group by industry and calculate sum of weights
            sum_weights_industry = df_today.groupby('industry')[weight_name].sum()
            industry_weight_sum_series = pd.Series(sum_weights_industry.values, index=sum_weights_industry.index)
            return industry_weight_sum_series

        def get_weighted_avg_indus_mkt_val(df, weight_name, date):           
            df_temp = df.loc[date].copy()  
            df_today = df_temp.loc[df_temp[weight_name] > 0].copy()

            # Group by industry and calculate sum of weights
            df_today['sum_weights_industry'] = df_today.groupby('industry')[weight_name].transform('sum')
            
            # Calculate industry weights
            df_today['indus_weight'] = df_today[weight_name] / df_today['sum_weights_industry']
            # Calculate weighted market value average
            df_today.loc[:, 'weighted_mkt_val_avg'] = df_today['indus_weight'] * df_today['mkt_val']
            # print(f'{weight_name}: \n', df_today)
            # Group by industry and sum the weighted market values
            indus_mkt_avg_sum = df_today.groupby('industry')['weighted_mkt_val_avg'].sum()
            # Create a Series from the summed values
            indus_mkt_val_weighted_avg = pd.Series(indus_mkt_avg_sum.values, index=indus_mkt_avg_sum.index)
            return indus_mkt_val_weighted_avg


        def weights_adjust_by_industry_mkt_val_alignment(zz1000, hs300, min400):
            all_industries = zz1000.index.union(hs300.index).union(min400.index)
            # Dictionary to store the results
            x_values = {}

            for industry in all_industries:
                zz1000_weight = zz1000.get(industry)
                hs300_weight = hs300.get(industry)
                min400_weight = min400.get(industry)

                if zz1000_weight is not None:
                    if hs300_weight is not None and min400_weight is not None:
                        # Scenario 1: This industry is in all 3 input series
                        if min400_weight != hs300_weight:
                            x = (zz1000_weight - min400_weight) / (hs300_weight - min400_weight)
                            if x > 1: # also can't do market val alignment; e.g. 申万环保：1.048729e+10, NaN, 1.433967e+09
                                x = -10 
                    elif hs300_weight is not None:
                        # Scenario 2: This industry is in hs300 and zz1000, but not in min400
                        x = zz1000_weight / hs300_weight
                    elif min400_weight is not None:
                        # Scenario 3: This industry is in min400 and zz1000, but not in hs300
                        x = zz1000_weight / min400_weight
                    else:
                        # Scenario 4: This industry is only in zz1000
                        x = -10  # Indicate that the portfolio should choose the constituents in zz1000 itself
                        # TODO: how to assign their weights?
        
                    x_values[industry] = x
            return x_values

        def phase1_weight_zoom(x_values, df_chosen_stocks_today_adjust_info, \
                                chosen_today_stocks, zz1000_indus_weights_sum_today, \
                                hs300_indus_weights_sum_today, min400_indus_weights_sum_today, init_weights):

            all_industries = x_values.keys()
            theory_weights = init_weights

            for industry in all_industries:
                # Calculate the weight sum of each industry
                zz1000_weight_sum = zz1000_indus_weights_sum_today.get(industry)
                hs300_weight_sum = hs300_indus_weights_sum_today.get(industry)
                min400_weight_sum = min400_indus_weights_sum_today.get(industry)
                x = x_values.get(industry)

                if x == -10:
                    x
                else:
                    # Calculate the target adjust weight for hs300 and min400
                    hs300_tar_weight = x * zz1000_weight_sum
                    min400_tar_weight = (1 - x) * zz1000_weight_sum
                
                    # Calculate the weight zooming scale: how much should we zoom the initial weights
                    hs300_zoom_scale = hs300_tar_weight / hs300_weight_sum
                    min400_zoom_scale = min400_tar_weight / min400_weight_sum

                    # Conduct weight zooming on initial weights
                    for stock_code in chosen_today_stocks:
                        if stock_industry == industry:
                            initial_weight = theory_weights.loc[stock_code]
                            hs300_adjusted_weight = initial_weight * hs300_zoom_scale

                            min400_adjusted_weight = initial_weight * min400_zoom_scale
                            # Assign adjusted weights back to theory_weights
                            theory_weights.loc[stock_code] = min(hs300_adjusted_weight, min400_adjusted_weight)

                
            return theory_weights

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
        

        # Compare with yesterday's stocks and get the three partition
        # remain_stocks = self.yesterday_stocks.intersection(today_stocks)
        # new_stocks = today_stocks - self.yesterday_stocks
        # delete_stocks = self.yesterday_stocks - today_stocks

        df_all_stocks_today_adjust_info = self.df_adjust_info.loc[date]

        initial_weights_today = pd.Series(0, index=weights.index)
        theory_weights_phase1 = pd.Series(0, index=weights.index)
        theory_weights_phase2 = pd.Series(0, index=weights.index)
        final_weights = pd.Series(0, index=weights.index)

        # Extract and update initial weights to the initial_weight_today series
        df_theory_zz1000_today = self.df_zz1000.loc[date].copy()
        df_theory_hs300_today = self.df_hs300.loc[date].copy()  
        df_theory_min400_today = self.df_theory_min400.loc[date].copy()  

        initial_hs300_weights = df_theory_hs300_today['hs300_weights']
        initial_min400_weights = df_theory_min400_today['min400_weights']
        initial_weights_today.update(initial_hs300_weights)
        initial_weights_today.update(initial_min400_weights)

        #////////////////////////////////////////////////////////////////////////////////////////////

        # Get the lists of theory stocks holding
        theory_zz1000_today_stocks_list = list(df_theory_zz1000_today.index.get_level_values(0))
        theory_hs300_today_stocks_list = list(df_theory_hs300_today.index.get_level_values(0))
        theory_min400_today_stocks_list = list(df_theory_min400_today.index.get_level_values(0))
        chosen_zz1000_today_stocks_list = list(df_theory_zz1000_today[df_theory_zz1000_today['zz1000_weights'] > 0].index.get_level_values(0))
        chosen_hs300_today_stocks_list = list(df_theory_hs300_today[df_theory_hs300_today['hs300_weights'] > 0].index.get_level_values(0))
        chosen_min400_today_stocks_list = list(df_theory_min400_today[df_theory_min400_today['min400_weights'] > 0].index.get_level_values(0))
        theory_today_stocks_list = theory_zz1000_today_stocks_list + theory_hs300_today_stocks_list + theory_min400_today_stocks_list
        chosen_today_stocks_list = chosen_zz1000_today_stocks_list + chosen_hs300_today_stocks_list + chosen_min400_today_stocks_list

        df_chosen_stocks_today_adjust_info = df_all_stocks_today_adjust_info.loc[df_all_stocks_today_adjust_info.index.isin(chosen_today_stocks_list)]
        # ///////////////////////////////////////////////////////////////////////////////////////////
        zz1000_indus_weighted_avg = get_weighted_avg_indus_mkt_val(self.df_zz1000, 'zz1000_weights', date)
        hs300_indus_weighted_avg = get_weighted_avg_indus_mkt_val(self.df_hs300, 'hs300_weights', date)
        min400_indus_weighted_avg = get_weighted_avg_indus_mkt_val(self.df_theory_min400, 'min400_weights', date)

        zz1000_weights_sum = calculate_industry_weight_sum(self.df_zz1000, 'zz1000_weights', date)
        hs300_weights_sum = calculate_industry_weight_sum(self.df_hs300, 'hs300_weights', date)
        min400_weights_sum = calculate_industry_weight_sum(self.df_theory_min400, 'min400_weights', date)

        x_values = weights_adjust_by_industry_mkt_val_alignment(zz1000_indus_weighted_avg, hs300_indus_weighted_avg, \
                                                                                            min400_indus_weighted_avg)
        
        theory_weights_phase1 = phase1_weight_zoom(x_values, df_chosen_stocks_today_adjust_info, chosen_today_stocks_list, \
                                                    zz1000_weights_sum, hs300_weights_sum, min400_weights_sum, initial_weights_today)
        # ////////////////////////////////////////////////////////////////////////////////////////////
        # Initialize the old_weights in the first day
        if date == self.start_day:
            self.old_weights = weights
            self.old_weights[:] = 0

        final_weights = suspend_limit_adjust(df_all_stocks_today_adjust_info, theory_min400_today_stocks_list, \
                                                self.old_weights, theory_weights_phase1, final_weights)
        
        

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












        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
        # Compare with yesterday's stocks and get the three partition
        # remain_stocks = self.yesterday_stocks.intersection(today_stocks)
        # new_stocks = today_stocks - self.yesterday_stocks
        # delete_stocks = self.yesterday_stocks - today_stocks
        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#