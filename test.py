import math
import torch
import torch.nn as nn
import configparser
import os
import pandas as pd
from datetime import date
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from Network.discriminator import Discriminator
from Network.simple_predictor import Simple_Predictor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from dataset import SensorSignalDataset
from utils import train_test_dataloader

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

# gpu-used
CUDA_DEVICES = torch.device("cuda:"+config.get('default', 'cuda_device_number') if torch.cuda.is_available() else "cpu")

synthetic_dataset_dir = config.get('GenTstVis', 'syntheticDataset_path') + '/' + config.get('GenTstVis', 'date_dir') + '/' + config.get('GenTstVis', 'classification_dir')

real_dataset_dir = config.get('GenTstVis', 'Dataset_path')


d_num_epochs = config.getint('test', 'd_num_epochs')
p_num_epochs = config.getint('test', 'p_num_epochs')
batch_size = config.getint('test', 'batch_size')
learning_rate = config.getfloat('test', 'learning_rate')

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


def Discriminative(real_train_data_loader, real_test_data_loader, synthetic_train_data_loader, synthetic_test_data_loader):

    # model
    model = Discriminator(module = 'gru',
                            time_stamp = 82,
                            input_size = 27,
                            hidden_dim = 108,
                            output_dim = 1,
                            num_layers = 2)
    model = model.to(CUDA_DEVICES)
    model.train()

    # loss
    critertion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print("Start Discriminator Training")


    for epoch in range(d_num_epochs):

        training_loss = 0.0

        synthetic_train_data_loader_iterator = iter(synthetic_train_data_loader)

        num_examples = 0

        for i, real_inputs in enumerate(real_train_data_loader):

            optimizer.zero_grad()

            fake_inputs = next(synthetic_train_data_loader_iterator)

            if fake_inputs == None:
                break

            fake_inputs = fake_inputs[0].to(CUDA_DEVICES)

            real_inputs = real_inputs[0].to(CUDA_DEVICES)

            real_outputs = model(real_inputs, None)
            fake_outputs = model(fake_inputs, None)

            fake_label = torch.zeros_like(fake_outputs)
            real_label = torch.ones_like(real_outputs)

            outputs = torch.cat((real_outputs, fake_outputs), 0)

            labels = torch.cat((real_label, fake_label), 0)

            D_loss = critertion(outputs, labels)

            D_loss.backward()
            optimizer.step()

            training_loss += D_loss.item() * real_inputs.size(0)

            num_examples += real_inputs.size(0)

        training_loss = training_loss / num_examples

        if epoch % (np.round(d_num_epochs / 5)) == 0:
            print('step: '+ str(epoch) + '/' + str(d_num_epochs) + ', d_loss: ' + str(np.round(training_loss, 4)))

    print("Finish Discriminator Training")

    model.eval()

    print("Start Discriminator Testing")

    synthetic_test_data_loader_iterator = iter(synthetic_test_data_loader)

    correct_results_sum = 0
    results_sum = 0

    with torch.no_grad():

        for i, real_inputs in enumerate(real_test_data_loader):

            fake_inputs = next(synthetic_test_data_loader_iterator)

            if fake_inputs == None:
                break

            fake_inputs = fake_inputs[0].to(CUDA_DEVICES)

            real_inputs = real_inputs[0].to(CUDA_DEVICES)

            real_output = model(real_inputs, None)
            fake_output = model(fake_inputs, None)

            fake_label = torch.zeros_like(fake_output)
            real_label = torch.ones_like(real_output)

            outputs = torch.cat((real_output, fake_output), 0)

            outputs = torch.round(torch.sigmoid(outputs))

            labels = torch.cat((real_label, fake_label), 0)

            correct_results_sum += (labels == outputs).sum().item()

            results_sum += (labels.shape[0] * labels.shape[1])
        
    acc = np.round((correct_results_sum / results_sum), 4)
        

    print("Finish Discriminator Testing")

    print("[test.py] Accuracy: {} %".format(acc* 100))
    discriminative_score = np.abs(0.5-acc)

    return discriminative_score



def Predictive(synthetic_train_data_loader, real_test_data_loader):

    # model
    model = Simple_Predictor()
    model = model.to(CUDA_DEVICES)
    model.train()

    # loss
    critertion = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print("Start Predictive Training")

    for epoch in range(p_num_epochs):

        training_loss = 0.0

        num_examples = 0

        for i, inputs in enumerate(synthetic_train_data_loader):

            optimizer.zero_grad()

            bat, seq, dim = inputs[0].shape

            # X = inputs[0][:, :-1, :(dim-1)]
            X = inputs[0][:, :-1, :]
            # print("X: {}".format(X.shape))

            # Y = inputs[0][:, 1:, (dim-1)].detach().clone()
            Y = inputs[0][:, seq-1, :].detach().clone()
            # print("Y: {}".format(Y.shape))

            X = X.to(CUDA_DEVICES)
            Y = Y.to(CUDA_DEVICES)
            
            y_pred = model(X, None)
            # print("y_pred: {}".format(y_pred.shape))

            loss = critertion(y_pred, Y)

            loss.backward()
            optimizer.step()

            training_loss += loss.item() * X.size(0)

            num_examples += X.size(0)

        training_loss = training_loss / num_examples

        if epoch % np.round(p_num_epochs / 5) == 0:
            print('step: '+ str(epoch) + '/' + str(p_num_epochs) + ', p_loss: ' + str(np.round(training_loss, 4)))

    print("Finish Predictive Training")

    model.eval()

    print("Start Predictive Testing")

    # Compute the performance in terms of MAE
    sum_absolute_errors = 0
    sum_examples = 0
    
    with torch.no_grad():

        for i, inputs in enumerate(real_test_data_loader):

            bat, seq, dim = inputs[0].shape

            X = inputs[0][:, :-1, :]

            Y = inputs[0][:, seq-1, :]

            X = X.to(CUDA_DEVICES)
            Y = Y.to(CUDA_DEVICES)
                
            Y_pred = model(X, None)

            sum_absolute_errors += torch.abs(torch.sum(torch.sub(Y_pred, Y))).item()

            sum_examples += Y.shape[0] * Y.shape[1]

    predictive_score = sum_absolute_errors / sum_examples

    print("Finish Predictive Testing")

    return predictive_score




if __name__ == '__main__':

    real_train_data_loader, real_test_data_loader = train_test_dataloader(dataset_dir=real_dataset_dir, mode='test')

    synthetic_train_data_loader, synthetic_test_data_loader = train_test_dataloader(dataset_dir=synthetic_dataset_dir, mode='test')

    discriminative_score = Discriminative(real_train_data_loader, real_test_data_loader, synthetic_train_data_loader, synthetic_test_data_loader)

    predictive_score = Predictive(synthetic_train_data_loader, real_test_data_loader)

    print("Discriminative score: {}".format(discriminative_score))
    print("Predictive score: {}".format(predictive_score))