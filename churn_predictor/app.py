from __future__ import annotations

import pandas as pd
import polars as pl

from .helpers import Data, Models

import sys

clients = None
file_path = None
save = True
if '--clients' in sys.argv:
    clients = eval(sys.argv[sys.argv.index('--clients') + 1])
if '--file' in sys.argv:
    file_path = sys.argv[sys.argv.index('--file') + 1]
if '--save' in sys.argv:
    save = sys.argv[sys.argv.index('--save') + 1]


def main(clients: list[int] = clients, file_path: str = file_path, save: bool = save) -> list[float]:
    if clients is None and file_path is None:
        raise ValueError('You need to provide a client or a file path pointing to a file with data.')
    if file_path is not None:
        clients = pl.read_parquet(file_path)
    # Loading the data
    df = Data.load_all_data()
    df = Data.add_features(df)
    # Loading the model
    model = Models.load_model()
    predictions = Models.predict_clients(clients, df, model)
    if save:
        data_folder = Data.find_folder('data')
        pd.DataFrame({'client_id': clients, 'churn_prediction': predictions}).to_csv(f'{data_folder}/predictions.csv',
                                                                                     index=False)
    return predictions


if __name__ == '__main__':
    SystemExit(main())
