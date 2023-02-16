from __future__ import annotations

from dateutil import parser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from collections import defaultdict
import os
import sys
import time
from typing import Union
import warnings

warnings.simplefilter('ignore', FutureWarning)

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

pl.Config.set_tbl_cols(2000)
pl.Config.set_tbl_rows(10)


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
                              pl.col('branch_id').cast(pl.UInt16),
                              pl.col('order_channel').cast(pl.Categorical)])
        return df

    @classmethod
    def load_client_data(cls, file_path: str = None) -> pl.DataFrame:
        if file_path is None:
            file_path = cls.find_file('sales_client_relationship_dataset.csv')
        df = pl.read_csv(file_path)
        df = df.with_columns([pl.col('client_id').cast(pl.UInt32),
                              pl.col('quali_relation').cast(pl.Categorical)])
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
                               .alias(col_name).cast(pl.UInt16)])

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
    def calculate_days_since_last_order(cls, df: pl.DataFrame, ref_date: str = '2019-09-22') -> pl.DataFrame:
        """This method calculates the number of days between the last order of a client and the ref_date."""
        return df.with_columns([(parser.parse(ref_date).date() - pl.col('date_order').max()).dt.days().cast(pl.Int16)
                               .over('client_id')
                               .alias('days_since_last_order')])

    @staticmethod
    def calculate_days_with_order(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([pl.n_unique('date_order').over('client_id').alias('days_with_order').cast(pl.UInt16)])

    @staticmethod
    def calculate_order_count(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([pl.count().over('client_id').alias('order_count')])

    @staticmethod
    def add_churn(df: pl.DataFrame, threshold: float = 1.5) -> pl.DataFrame:
        """This method creates a column with a value of 1 if the client is assumed to have churned and 0 otherwise.
        Churn is defined based on the time since the last order and the order frequency."""
        return df.with_columns(pl.when(pl.col('days_since_last_order') > threshold * pl.col('q100_order_freq'))
                               .then(1)
                               .otherwise(0).alias('churned'))

    @classmethod
    def add_features(cls, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(cls.calculate_item_price) \
            .pipe(cls.calculate_mean_price) \
            .pipe(cls.calculate_quantile_price) \
            .pipe(cls.calculate_mean_order_freq) \
            .pipe(cls.calculate_quantile_order_freq) \
            .pipe(cls.calculate_relative_price) \
            .pipe(cls.calculate_relative_price_client_specific) \
            .pipe(cls.calculate_days_since_last_order) \
            .pipe(cls.calculate_days_with_order) \
            .pipe(cls.calculate_order_count) \
            .pipe(cls.calculate_quantile_order_freq, quantile=1, col_name='q100_order_freq') \
            .pipe(cls.add_churn, threshold=1.5)

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


class Plots:
    @staticmethod
    def format_label(label: str, max_len: int = 20) -> str:
        line = ''
        lines = []
        for word in label.split(' '):
            if len(line) + len(word) <= max_len:
                line += word + ' '
            else:
                lines += [line]
                line = word + ' '
        lines += [line.strip()]
        return '\n'.join(lines)

    @classmethod
    def hist_plot(cls, df: Union[pd.DataFrame, pl.DataFrame], x: str, y: str, agg_func: str = 'mean',
                  save: bool = False, file_name: str = None) -> None:
        """This method creates a histogram with x on the x-axis and y on y-axis, agg_func defines how values should be
        aggregated if there are multiple y values for an x value."""
        if type(df) == pl.DataFrame:
            data = df.groupby(x).agg(eval(f"pl.{agg_func}('{y}')")).to_pandas()
        elif type(df) == pd.DataFrame:
            data = df.copy()
        else:
            raise TypeError('The df can only be a pandas or a polars DataFrame.')

        plot = sns.barplot(data=data, x=x, y=y, estimator=agg_func)
        plot.set_xticklabels([cls.format_label(label) for label in data[x]])
        plt.tight_layout()
        if save:
            if file_name is None:
                file_name = f'barplot_{x}_{y}.jpeg'
            plot.get_figure().savefig(file_name)


# df = DataPL.load_all_data()
# df = DataPL.add_features(df)
