# -*- coding: utf-8 -*-
from dataset_preprocess import preprocess
import torch
from torch.utils.data import Dataset
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

def StandardScaler(data):
    """Min Max normalizer.
    do for each column
    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    m = data.mean(0, keepdim=True)
    s = data.std(0, unbiased=False, keepdim=True)
    data = data - m
    # epsilon = 1e-7 to avoid loss=nan
    norm_data = data / (s + 1e-7)
    return norm_data

def MinMaxScaler1(data):
  """Min Max normalizer.
  do for each column
  Args:
    - data: original data

  Returns:
    - norm_data: normalized data
  """
  min_val = np.min(data, 0)
  max_val = np.max(data, 0)
  numerator = data - min_val
  denominator = max_val - min_val
  norm_data = numerator / (denominator + 1e-7)
  # rescale to (-1, 1)
  norm_data = 2 * norm_data - 1
  return norm_data, min_val, max_val

def MinMaxScaler2(data):
    """Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val

    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
    #[test] min-max to (-1, 1)
    norm_data = 2 * norm_data - 1

    return norm_data, min_val, max_val


class SensorSignalDataset(Dataset):
    # root_dir:資料集路徑
    # mode:有'train'、'test'兩種，依據使用需要來選擇取用訓練資料集、測試資料集
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.path = preprocess(root_dir)

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):

        ori_data = np.loadtxt(self.path[index], delimiter = ",",skiprows = 0)
        ori_data = np.around(ori_data, decimals=4)

        ori_data = ori_data[::-1]

        ori_data, min_val1, max_val1 = MinMaxScaler1(ori_data)
        data, min_val2, max_val2 = MinMaxScaler2(ori_data)

        data = torch.FloatTensor(data)
        data = np.around(data, decimals=4)

        return data, min_val1, max_val1, min_val2, max_val2
