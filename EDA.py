from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from collections import defaultdict
import os
import sys
import time
import warnings

warnings.simplefilter('ignore', FutureWarning)

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

pl.Config.set_tbl_cols(2000)


# # Reading in the data
# df = pd.read_parquet('transaction_data.parquet')
# df = pd.merge(df, pd.read_csv('sales_client_relationship_dataset.csv'), on='client_id', how='inner')
# # Adjusting the data types
# df.loc[:, ['date_order', 'date_invoice']] = df.loc[:, ['date_order', 'date_invoice']].astype('datetime64')
# df.loc[:, 'quantity'] = df.loc[:, 'quantity'].astype('int32')

# # Grouping by clients
# client_group = df.groupby('client_id')['sales_net'].sum().sort_values()
# # Most clients have positive net sales -> negative net sales can't be us buying from the suppliers
# (client_group < 0).sum()
# # 1296 clients have more than 10^6 net sales
# (client_group > 10**6).sum()
# # One percent of clients is responsible for 1/3 of the net sales
# client_group.iloc[-1706:].sum() / client_group.sum()


class Data:
    @staticmethod
    def find_file(file_name: str) -> str:
        for root, folders, files in os.walk('.'):
            if file_name in files:
                return root + '/' + file_name

    @classmethod
    def load_transactions(cls, file_path: str = None) -> pd.DataFrame:
        if file_path is None:
            file_path = cls.find_file('transaction_data.parquet')
        df = pd.read_parquet(file_path)
        # Don't use .loc here, since the garbage collector won't remove the copies
        df[['date_order', 'date_invoice']] = df[['date_order', 'date_invoice']].astype('datetime64')
        df[['quantity', 'client_id', 'product_id', 'branch_id']] = \
            df[['quantity', 'client_id', 'product_id', 'branch_id']].astype('int32')
        return df

    @classmethod
    def load_client_data(cls, file_path: str = None) -> pd.DataFrame:
        if file_path is None:
            file_path = cls.find_file('sales_client_relationship_dataset.csv')
        df = pd.read_csv(file_path)
        df['client_id'] = df['client_id'].astype('int32')
        return df

    @classmethod
    def load_all_data(cls, transactions_file: str = None, client_file: str = None) -> pd.DataFrame:
        return pd.merge(cls.load_transactions(transactions_file), cls.load_client_data(client_file),
                        on='client_id', how='inner')

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        return df[df['sales_net'] > 0]

    @staticmethod
    def calculate_item_price(df: pd.DataFrame) -> pd.DataFrame:
        df['item_price'] = df['sales_net'] / df['quantity']
        return df


class DataPL:
    @staticmethod
    def find_file(file_name: str) -> str:
        for root, folders, files in os.walk('.'):
            if file_name in files:
                return root + '/' + file_name

    @classmethod
    def load_transactions(cls, file_path: str = None) -> pl.DataFrame:
        if file_path is None:
            file_path = cls.find_file('transaction_data.parquet')
        df = pl.read_parquet(file_path)
        df = df.with_columns([pl.col(['date_order', 'date_invoice']).str.strptime(datatype=pl.Date),
                              pl.col(['quantity', 'client_id', 'product_id']).cast(pl.UInt32),
                              pl.col('branch_id').cast(pl.UInt16)])
        return df

    @classmethod
    def load_client_data(cls, file_path: str = None) -> pl.DataFrame:
        if file_path is None:
            file_path = cls.find_file('sales_client_relationship_dataset.csv')
        df = pl.read_csv(file_path)
        df = df.with_columns([pl.col('client_id').cast(pl.UInt32)])
        return df

    @classmethod
    def load_all_data(cls, transactions_file: str = None, client_file: str = None, clean: bool = True) -> pl.DataFrame:
        df_merged = cls.load_transactions(transactions_file).join(cls.load_client_data(client_file),
                                                                  on='client_id', how='inner')
        if clean:
            df_merged = cls.clean_data(df_merged)
        return df_merged

    @staticmethod
    def clean_data(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter((pl.col('sales_net') >= 0) & (pl.col('date_invoice') >= pl.col('date_order')))

    @staticmethod
    def calculate_item_price(df: pl.DataFrame) -> pl.DataFrame:
        # Vectorized operations are the same speed in polars and pandas
        return df.lazy().with_columns([(pl.col('sales_net') / pl.col('quantity')).alias('item_price')]).collect()

    @classmethod
    def calculate_mean_price(cls, df: pl.DataFrame) -> pl.DataFrame:
        if 'item_price' not in df.columns:
            df = cls.calculate_item_price(df)
        return df.with_columns([pl.mean('item_price').over('product_id').alias('mean_item_price')])

    @classmethod
    def calculate_quantile_price(cls, df: pl.DataFrame, quantile: float = 0.5,
                                 col_name: str = 'median_item_price') -> pl.DataFrame:
        if 'item_price' not in df.columns:
            df = cls.calculate_item_price(df)
        return df.with_columns([pl.quantile('item_price', quantile).over('product_id').alias(col_name)])

    @staticmethod
    def calculate_mean_order_freq(df: pl.DataFrame) -> pl.DataFrame:
        """This method calculates how often (on average) a client orders something"""
        return df.with_columns([((pl.max('date_order') - pl.min('date_order')).dt.days()
                                 / pl.n_unique('date_order'))
                               .over('client_id')
                               .alias('mean_order_freq')])

    @staticmethod
    def calculate_quantile_order_freq(df: pl.DataFrame, quantile: float = 0.5,
                                      col_name: str = 'median_order_freq') -> pl.DataFrame:
        """This method calculates the quantile frequency of a client's orders."""
        return df.with_columns([pl.col('date_order').unique().sort().diff(1).dt.days().quantile(quantile)
                               .over('client_id')
                               .alias(col_name)])

    @staticmethod
    def calculate_mean_price_unbiased(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([(pl.sum('sales_net') / pl.sum('quantity'))
                               .over('product_id')
                               .alias('unbiased_mean_price')])

    @classmethod
    def calculate_mean_price_client_specific(cls, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([(pl.sum('sales_net') / pl.sum('quantity'))
                               .over(['product_id', 'client_id'])
                               .alias('unbiased_client_specific_mean_price')])

    @classmethod
    def calculate_relative_price(cls, df: pl.DataFrame) -> pl.DataFrame:
        if 'unbiased_mean_price' not in df.columns:
            df = cls.calculate_mean_price_unbiased(df)
        return df.with_columns([((pl.col('item_price') - pl.col('unbiased_mean_price'))
                                 / pl.col('unbiased_mean_price'))
                               .alias('relative_price')])

    @classmethod
    def calculate_relative_price_client_specific(cls, df: pl.DataFrame) -> pl.DataFrame:
        if 'unbiased_client_specific_mean_price' not in df.columns:
            df = cls.calculate_mean_price_client_specific(df)
        return df.with_columns([((pl.col('item_price') - pl.col('unbiased_client_specific_mean_price'))
                                 / pl.col('unbiased_client_specific_mean_price'))
                               .alias('client_specific_relative_price')])

    @classmethod
    def add_features(cls, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(cls.calculate_item_price) \
            .pipe(cls.calculate_mean_price) \
            .pipe(cls.calculate_quantile_price) \
            .pipe(cls.calculate_mean_order_freq) \
            .pipe(cls.calculate_quantile_order_freq) \
            .pipe(cls.calculate_relative_price) \
            .pipe(cls.calculate_relative_price_client_specific)

    @classmethod
    def load_data_for_model(cls) -> pl.DataFrame:
        df = cls.load_all_data().pipe(cls.add_features())
        return df.drop(['date_order',
                        'date_invoice',
                        'sales_net',
                        'quantity',
                        'order_channel',
                        'quality_relation',
                        'item_price',
                        'mean_item_price',
                        'median_item_price',
                        'mean_order_freq',
                        'median_order_freq',
                        'unbiased_mean_price',
                        'relative_price',
                        'unbiased_client_specific_mean_price',
                        'client_specific_relative_price'])


class Analyzer:
    @staticmethod
    def get_percent_of_clients_for_income_quantile(df: pd.DataFrame, quantile: float = 0.8) -> float:
        """This function returns the fraction of clients that are responsible for the quantile of the total income."""
        client_group = df.groupby('client_id').sales_net.sum().sort_values(ascending=False)
        target_sum = quantile * client_group.sum()
        iter_sum = 0
        last_index = 0
        for index, value in enumerate(client_group):
            if iter_sum < target_sum:
                iter_sum += value
                last_index = index
        return last_index / len(client_group) * 100

    @staticmethod
    def get_percent_of_income_for_client_quantile(df: pd.DataFrame, quantile: float = 0.1) -> float:
        """This function returns the fraction of clients that are responsible for the quantile of the total income."""
        client_group = df.groupby('client_id').sales_net.sum().sort_values(ascending=False)
        index_of_quantile = int(quantile * len(client_group))
        return client_group.iloc[:index_of_quantile].sum() / client_group.sum() * 100


df = DataPL.load_all_data()
df = DataPL.add_features(df)
