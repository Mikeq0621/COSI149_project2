import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.model import batch_size, seq_length, output_size

def from_file_to_data_loader(price_f, nas_f):
    raw_df = pd.read_csv(price_f, parse_dates=['Date']).sort_values(by='Date')
    raw_df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'])
    # NASDAQ index
    nas_df = pd.read_csv(nas_f, parse_dates=['Date']).sort_values(by='Date')

    # normalize close price
    cp = raw_df['Close'].to_numpy().reshape(-1, 1)
    cp_scaler = MinMaxScaler().fit(cp)
    norm_scale = cp_scaler.transform(cp)
    df = pd.DataFrame(norm_scale, columns=['Close'])
    # Normalize nasdaq index
    nas = nas_df['Close'].to_numpy().reshape(-1, 1)
    nas_scaler = MinMaxScaler().fit(nas)
    nas_norm_scaler = nas_scaler.transform(nas)
    nas_df = pd.DataFrame(nas_norm_scaler, columns=['Close'])
    # Normalize Volume
    vol = raw_df['Volume'].to_numpy().reshape(-1, 1)
    vol_scaler = MinMaxScaler().fit(vol)
    vol_norm_scaler = vol_scaler.transform(vol)
    vol_df = pd.DataFrame(vol_norm_scaler, columns=['Volume'])

    df['prev1'] = df['Close'].shift(1)
    df['prev2'] = df['Close'].shift(2)
    df['prev3'] = df['Close'].shift(3)
    df['prev4'] = df['Close'].shift(4)
    df['prev5'] = df['Close'].shift(5)

    df['nas1'] = nas_df['Close'].shift(1)
    df['nas2'] = nas_df['Close'].shift(2)
    df['nas3'] = nas_df['Close'].shift(3)
    df['nas4'] = nas_df['Close'].shift(4)
    df['nas5'] = nas_df['Close'].shift(5)
    df['vol1'] = vol_df['Volume'].shift(1)
    df['vol2'] = vol_df['Volume'].shift(2)
    df['vol3'] = vol_df['Volume'].shift(3)
    df['vol4'] = vol_df['Volume'].shift(4)
    df['vol5'] = vol_df['Volume'].shift(5)
    df['avg'] = (df['prev1'] + df['prev2'] + df['prev3'] + df['prev4'] + df['prev5']) / 5.0

    df = df.dropna(subset=['prev1', 'prev2', 'prev3', 'prev4', 'prev5', 'nas1', 'nas2', 'nas3', 'nas4', 'nas5',
                           'vol1', 'vol2', 'vol3', 'vol4', 'vol5', 'avg'
                           ]).reset_index(drop=True)
    x = df[['prev1', 'prev2', 'prev3', 'prev4', 'prev5', 'nas1', 'nas2', 'nas3', 'nas4', 'nas5',
            'vol1', 'vol2', 'vol3', 'vol4', 'vol5', 'avg']].to_numpy(dtype=np.float32)
    # df = df.dropna(subset=['prev1', 'prev2', 'prev3', 'prev4', 'prev5']).reset_index(drop=True)
    # x = df[['prev1', 'prev2', 'prev3', 'prev4', 'prev5']].to_numpy(dtype=np.float32)

    y = df[['Close']].to_numpy(dtype=np.float32).reshape(-1, 1)
    print(str(x.shape) + ' ' + str(y.shape) + '\n')

    xs = []
    ys = []
    for i in range(0, len(x) - seq_length + 1, 1):
        xs.append(x[i: i + seq_length, :].copy())
        ys.append(y[i: i + seq_length, :].copy())
    xs = np.array(xs)
    ys = np.array(ys)

    xs = []
    ys = []
    for i in range(0, len(x) - seq_length + 1, 1):
        xs.append(x[i: i + seq_length, :].copy())
        ys.append(y[i: i + seq_length, :].copy())
    xs = np.array(xs)
    ys = np.array(ys)
    print("train data shape: " + str(xs.shape) + ' ' + str(ys.shape) + '\n')

    x_predict, y_predict= xs, ys

    x_predict = torch.from_numpy(x_predict)

    y_predict = torch.from_numpy(y_predict)

    train_dataset = Data.TensorDataset(x_predict, y_predict)

    data_loader = Data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False# random shuffle for training
    )
    return data_loader, cp_scaler

def read_last_item(file, column='Close'):
    raw_df = pd.read_csv(file, parse_dates=['Date']).sort_values(by='Date')
    cp = raw_df[column].to_numpy().reshape(-1, 1)
    return cp[-1]


