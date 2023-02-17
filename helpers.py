from __future__ import annotations

from dateutil import parser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rnns import RNN

import os
from typing import Any, Union
import warnings

warnings.simplefilter('ignore', FutureWarning)

# pl.Config.set_tbl_cols(2000)
# pl.Config.set_tbl_rows(10)


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
        return df.filter((pl.col('sales_net') > 0)
                         & (pl.col('date_invoice') >= pl.col('date_order'))
                         & (pl.col('date_order').n_unique().over('client_id') > 1))

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
                               .otherwise(0).alias('churned').cast(pl.Int8))

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

    @staticmethod
    def normalize(df: pl.DataFrame, columns: list[Union[str, int]]) -> pl.DataFrame:
        return df.with_columns([pl.col(col) / pl.max(col) for col in columns])

    @staticmethod
    def one_hot_encode(df: pl.DataFrame, columns: list[Union[str, int]]) -> pl.DataFrame:
        # index_cols = [col for col in df.columns if col not in columns]
        # df.with_columns(pl.lit(1).alias('one').cast(pl.Int8)) \
        #     .pivot(index=index_cols, columns=columns, values='one') \
        #     .fill_null(0)
        return df.to_dummies(columns=columns)

    @classmethod
    def load_data_for_model(cls, df: pl.DataFrame = None, keep_first_n_rows: int = None) -> pl.DataFrame:
        # Handling the input
        if df is None:
            df = cls.load_all_data().pipe(cls.add_features())
        if keep_first_n_rows is not None:
            df = df[:keep_first_n_rows]
        # Keep only relevant columns
        df = df[['sales_net',
                 'quantity',
                 'order_channel',
                 'quali_relation',
                 'item_price',
                 'mean_item_price',
                 'median_item_price',
                 'mean_order_freq',
                 'median_order_freq',
                 'unbiased_mean_price',
                 'relative_price',
                 'unbiased_client_specific_mean_price',
                 'client_specific_relative_price',
                 'days_with_order',
                 'order_count',
                 'q100_order_freq',
                 'days_since_last_order',
                 'churned']]
        # Preprocess
        numeric_cols = [df.columns[i] for i, t in enumerate(df.dtypes) if pl.datatypes.NumericType in t.mro()]
        numeric_cols = [col for col in numeric_cols if col != 'churned']
        df = cls.normalize(df, columns=numeric_cols)
        cat_cols = ['order_channel', 'quali_relation']
        df = cls.one_hot_encode(df, columns=cat_cols)
        return df

    @classmethod
    def load_client_data_for_model(cls, df: pl.DataFrame, client_ids: Union[int, list[int]]) -> pl.DataFrame:
        # To see all categories for the one-hot encoding, we need to have at least one record for each category
        # (1217701 used all five order channels and is agreeable, 1365398 is difficult, 450600 is demanding (these would
        # be the most data efficient ones) 188502 for online and difficult, 835089 for at the store and demanding,
        # 1977896 for agreeable, 2086861 for by phone, 1421553 for during the visit of a sales rep, 1277522 for other (
        # we have to use these, since the one-hot encoding is based on what categories are first found in the data)
        if type(client_ids) == int:
            client_ids = [client_ids]
        necessary_ids = [188502, 835089, 1977896, 2086861, 1421553, 1277522]
        df = df.filter(pl.col('client_id').is_in(necessary_ids + client_ids))
        mask = df['client_id'].is_in(client_ids)
        return cls.load_data_for_model(df).filter(mask).drop('churned')


class PyTorchDataset(Dataset):
    def __init__(self, df: pl.DataFrame = None, keep_first_n_rows: int = None) -> None:
        df = DataPL.load_data_for_model(df, keep_first_n_rows=keep_first_n_rows)

        self.X = df.drop(['churned'])
        self.y = df[['churned']].to_numpy()

    def __getitem__(self, item):
        return self.X[item].to_numpy(), self.y[item]

    def __len__(self):
        return self.X.shape[0]


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


class Models:
    # Defining device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Defining hyper-parameters
    num_epochs = 2
    batch_size = 64
    learning_rate = 10 ** -4
    num_layers = 2

    # Defining data shapes
    num_classes = 1
    input_size = 23
    hidden_size = 128

    # File name to save the model to
    file_name = 'model.pth'

    @classmethod
    def create_rnn(cls, input_size: int = None, hidden_size: int = None, num_layers: int = None,
                   num_classes: int = None) -> RNN:
        # Handling the input
        if input_size is None:
            input_size = cls.input_size
        if hidden_size is None:
            hidden_size = cls.hidden_size
        if num_layers is None:
            num_layers = cls.num_layers
        if num_classes is None:
            num_classes = cls.num_classes

        return RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                   num_classes=num_classes).float()

    @classmethod
    def train_model(cls, model: Any, df: pl.DataFrame, learning_rate: float = None, num_epochs: int = None,
                    batch_size: int = None, keep_first_n_rows: int = None,
                    save: bool = True, file_name: str = None) -> Any:
        # Handling input
        if learning_rate is None:
            learning_rate = cls.learning_rate
        if num_epochs is None:
            num_epochs = cls.num_epochs
        if batch_size is None:
            batch_size = cls.batch_size

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        dataset = PyTorchDataset(df, keep_first_n_rows=keep_first_n_rows)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        n_steps = len(data_loader)

        for epoch in range(num_epochs):
            for i, (X, y) in enumerate(data_loader):
                X = X.float().to(cls.device)
                y = y.float().to(cls.device)

                output = model(X)
                loss = criterion(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(f'Epoch {epoch + 1}/{num_epochs}, Step {i + 1}/{n_steps}, Loss: {loss.item():.4f}')
        cls.model = model
        if save:
            if file_name is None:
                file_name = cls.file_name
            torch.save(model.stat_dict(), file_name)
        return model

    @classmethod
    def load_model(cls, model_path: str = 'model.pth') -> Any:
        model = cls.create_rnn()
        model.load_state_dict(torch.load(model_path, map_location=cls.device))
        model = model.to(cls.device)
        return model.eval()

    @classmethod
    def predict(cls, X: torch.Tensor, model: Any = None) -> np.ndarray:
        if model is None:
            model = cls.model
        if X.ndim == 2:
            X = X.unsqueeze(0)

        with torch.no_grad():
            X = X.float().to(cls.device)
            return model(X).numpy()

    @classmethod
    def predict_client(cls, client_ids: Union[int, list[int]], df: pl.DataFrame, model: Any = None) -> np.ndarray:
        X = torch.from_numpy(DataPL.load_client_data_for_model(df, client_ids).to_numpy())
        X = X.unsqueeze(0)
        return cls.predict(X=X, model=model)


# df = DataPL.load_all_data()
# df = DataPL.add_features(df)
