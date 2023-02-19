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

from .rnns import RNN

import os
from typing import Any, Union


class Data:
    normalization_basis: pl.DataFrame = None
    one_hot_basis: pl.DataFrame = None

    @staticmethod
    def find_file(file_name: str) -> str:
        for root, folders, files in os.walk('.'):
            if file_name in files:
                return root + '/' + file_name

    @staticmethod
    def find_folder(folder: str) -> str:
        for root, folders, files in os.walk('.'):
            if folder in root:
                return root

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
    def load_all_data(cls, transactions_file: str = None, client_file: str = None, clean: bool = True,
                      order: bool = True, add_index: bool = True, add_range: bool = True) -> pl.DataFrame:
        df_merged = cls.load_transactions(transactions_file).join(cls.load_client_data(client_file),
                                                                  on='client_id', how='inner')
        if clean:
            df_merged = cls.clean_data(df_merged)
        if order:
            df_merged = cls.order_data(df_merged)
        if add_index:
            df_merged = cls.add_index(df_merged)
        if add_range:
            df_merged = cls.add_range(df_merged)
        return df_merged

    @classmethod
    def order_data(cls, df: pl.DataFrame) -> pl.DataFrame:
        return df.sort(by=['client_id', 'date_order', 'product_id'])

    @staticmethod
    def clean_data(df: pl.DataFrame) -> pl.DataFrame:
        sales_c = pl.col('sales_net') > 0
        date_c = pl.col('date_invoice') >= pl.col('date_order')
        order_c = pl.col('date_order').n_unique().over('client_id') > 1
        # Since the filter by one condition is executed independent of the other conditions, we have to chain the last
        # one
        return df.filter(sales_c & date_c).filter(order_c)

    @staticmethod
    def add_index(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.Series(name='index', values=np.arange(len(df))))

    @classmethod
    def add_range(cls, df: pl.DataFrame) -> pl.DataFrame:
        if 'index' not in df.columns:
            df = cls.add_index(df)
        return df.with_columns([pl.col('index').first().alias('first').over('client_id'),
                                pl.col('index').last().alias('last').over('client_id')])

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

    @classmethod
    def create_normalization_basis(cls, df: pl.DataFrame, columns: list[str], save: bool = False) -> pl.DataFrame:
        normalization_basis = df[columns].max()
        if save:
            cls.normalization_basis = normalization_basis
        return normalization_basis

    @classmethod
    def normalize(cls, df: pl.DataFrame, columns: list[Union[str, int]]) -> pl.DataFrame:
        if cls.normalization_basis is None:
            cls.create_normalization_basis(df, columns=columns, save=True)
        return df.with_columns([pl.col(col) / cls.normalization_basis[col][0] for col in columns])

    @classmethod
    def create_one_hot_basis(cls, df: pl.DataFrame, columns: list[str], save: bool = False) -> pl.DataFrame:
        bases = []
        for column in columns:
            bases.append(df[df.groupby(column).agg(pl.col('index').first())['index']])
        one_hot_basis = pl.concat(bases, how='vertical')
        if save:
            cls.one_hot_basis = one_hot_basis
        return one_hot_basis

    @classmethod
    def one_hot_encode(cls, df: pl.DataFrame, columns: list[Union[str, int]]) -> pl.DataFrame:
        # index_cols = [col for col in df.columns if col not in columns]
        # df.with_columns(pl.lit(1).alias('one').cast(pl.Int8)) \
        #     .pivot(index=index_cols, columns=columns, values='one') \
        #     .fill_null(0)
        if cls.one_hot_basis is None:
            cls.create_one_hot_basis(df, columns=columns, save=True)
        # Since we return the data without the one_hot_basis, rechunking is not needed which saves time
        df = pl.concat([cls.one_hot_basis, df], how='vertical', rechunk=False)
        return df.to_dummies(columns=columns)[len(cls.one_hot_basis):]

    @classmethod
    def load_data_for_model(cls, df: pl.DataFrame = None, keep_first_n_clients: int = None) -> pl.DataFrame:
        # Handling the input
        if df is None:
            df = cls.load_all_data().pipe(cls.add_features)
        # Defining the numerical and categorical columns
        numeric_cols = [df.columns[i] for i, t in enumerate(df.dtypes) if pl.datatypes.NumericType in t.mro()]
        not_to_scale = ['index', 'first', 'last', 'client_id', 'product_id', 'branch_id', 'churned']
        numeric_cols = [col for col in numeric_cols if col not in not_to_scale]
        cat_cols = ['order_channel', 'quali_relation']

        if keep_first_n_clients is not None:
            first_n_clients = df['client_id'].unique()[:keep_first_n_clients]
            # This makes sure that the scaling will be consistent. You can think of it as the .fit in scikit-learn
            cls.create_normalization_basis(df=df, columns=numeric_cols, save=True)
            # This makes sure that the encoding will be done along all categories. You can think of it as the .fit in
            # scikit-learn
            cls.create_one_hot_basis(df=df, columns=cat_cols, save=True)
            df = df.filter(pl.col('client_id').is_in(first_n_clients))
            df = cls.order_data(df)

        # Preprocess
        df = cls.one_hot_encode(df, columns=cat_cols)
        df = cls.normalize(df, columns=numeric_cols)
        # We can drop the columns that we don't need anymore
        df = df.drop(['index', 'date_order', 'date_invoice', 'product_id', 'branch_id', 'client_id'])
        return df


class PyTorchDataset(Dataset):
    def __init__(self, df: pl.DataFrame = None, keep_first_n_clients: int = None) -> None:
        self.ranges = df.groupby('client_id').agg([pl.col('first').first(),
                                                   pl.col('last').first()]).sort(by='client_id')
        self.ranges = self.ranges[:keep_first_n_clients]
        df = Data.load_data_for_model(df, keep_first_n_clients=keep_first_n_clients)
        df = df.drop(['first', 'last'])
        self.X = df.drop(['churned'])
        self.y = df[['churned']]

    def __getitem__(self, item):
        return (self.X[self.ranges[item, 'first']: self.ranges[item, 'last'] + 1].to_numpy(),
                self.y[self.ranges[item, 'first']].to_numpy())

    def __len__(self):
        return len(self.ranges)


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
    batch_size = 8
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
                    batch_size: int = None, keep_first_n_clients: int = None,
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

        dataset = PyTorchDataset(df, keep_first_n_clients=keep_first_n_clients)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        n_steps = len(data_loader)

        for epoch in range(num_epochs):
            for i, (X, y) in enumerate(data_loader):
                X = X.float().to(cls.device)
                y = y.float().to(cls.device)
                output = model(X)
                loss = criterion(output.unsqueeze(0), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(f'Epoch {epoch + 1}/{num_epochs}, Step {i + 1}/{n_steps}, Loss: {loss.item():.4f}')
        cls.model = model
        if save:
            if file_name is None:
                file_name = cls.file_name
            torch.save(model.state_dict(), file_name)
        return model

    @classmethod
    def load_model(cls, file_name: str = 'model.pth') -> Any:
        model = cls.create_rnn()
        file_path = Data.find_file(file_name=file_name)
        model.load_state_dict(torch.load(file_path, map_location=cls.device))
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
    def predict_clients(cls, client_ids: Union[int, list[int]], df: pl.DataFrame, model: Any = None) -> list[float]:
        """This method can be used to make predictions on specific clients."""
        if type(client_ids) == int:
            client_ids = [client_ids]
        if model is None:
            model = cls.model

        ranges = df.groupby('client_id').agg([pl.col('first').first(),
                                              pl.col('last').first()]).sort(by='client_id')
        df = Data.load_data_for_model(df)
        df = df.drop(['first', 'last'])
        X = df.drop(['churned'])

        predictions = []
        for client_id in client_ids:
            index = ranges.filter(pl.col('client_id') == client_id)
            predictions.append(cls.predict(X=torch.from_numpy(X[index['first'][0]: index['last'][0] + 1].to_numpy()),
                                           model=model).item())
        return predictions
