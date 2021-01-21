import torch
import torch.nn
from dataset import SensorSignalDataset
from torch.utils.data import DataLoader
import configparser
import random


config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

manualSeed = config.getint('default', 'manualSeed')

random.seed(manualSeed)
torch.manual_seed(manualSeed)

def random_generator(batch_size, seq_len, dim):

  Z = torch.zeros(batch_size, seq_len, dim)

  for i in range(batch_size):
    temp = torch.rand(seq_len, dim)
    temp = torch.add(torch.mul(temp, 2.0), -1.0)
    Z[i, :, :] = temp.detach().clone()

  return Z

def train_test_dataloader(dataset_dir="", mode='test'):

  data_set = SensorSignalDataset(root_dir=dataset_dir, transform=None)

  train_dataset_size = int(config.getfloat(mode, 'trainset_percentage') * len(data_set))
  test_dataset_size = len(data_set) - train_dataset_size
  train_dataset, test_dataset = torch.utils.data.random_split(data_set, [train_dataset_size, test_dataset_size])

  train_data_loader = DataLoader(dataset=train_dataset, batch_size=config.getint(mode, 'batch_size'), shuffle=True, num_workers=1)
  test_data_loader = DataLoader(dataset=test_dataset, batch_size=config.getint(mode, 'batch_size'), shuffle=True, num_workers=1)

  return train_data_loader, test_data_loader



if __name__ == '__main__':

  # Z = random_generator(32, 82, 34)
  # print(Z)

  real_dataset_dir = config.get('default', 'Dataset_path') + '/' + config.get('default', 'classification_dir')

  train_data_loader, test_data_loader = train_test_dataloader(real_dataset_dir, 0.75)

  for i, inputs in enumerate(train_data_loader):

    X = inputs[0]

    print("[utils.py] i: {}, data loader: {}".format(i, X.shape))