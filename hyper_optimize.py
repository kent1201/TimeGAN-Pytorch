import math
import configparser
import os
from datetime import date
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import itertools
from Network.embedder import Embedder
from Network.recovery import Recovery
from Network.supervisor import Supervisor
from Network.generator import Generator
from Network.discriminator import Discriminator
from dataset import SensorSignalDataset
from Loss.embedder_loss import EmbedderLoss
from Loss.supervised_loss import SupervisedLoss
from Loss.joint_Gloss import JointGloss
from Loss.joint_Eloss import JointEloss
from Loss.joint_Dloss import JointDloss
from utils import random_generator

from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

config = configparser.ConfigParser()
config.read('Configure.ini', encoding="utf-8")

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True


# gpu-used
CUDA_DEVICES = torch.device("cuda:"+config.get('default', 'cuda_device_number') if torch.cuda.is_available() else "cpu")
dataset_dir = config.get('train', 'Dataset_path') + '/' + config.get('train', 'classification_dir').split('_')[0]
seq_len = config.getint('train', 'seq_len')
n_features = config.getint('train', 'n_features')
classification_dir = config.get('train', 'classification_dir')
module_name = config.get('default', 'module_name')
embedder_name = config.get('default', 'embedder_name')
recovery_name = config.get('default', 'recovery_name')
generator_name = config.get('default', 'generator_name')
supervisor_name = config.get('default', 'supervisor_name')
discriminator_name = config.get('default', 'discriminator_name')

# 1. Embedding network training
def train_stage1(H_config, embedder, recovery):

  # Dataset
  data_set = SensorSignalDataset(root_dir=dataset_dir, transform=None)
  data_loader = DataLoader(dataset=data_set, batch_size=H_config["batch_size"], shuffle=False, num_workers=1)

  # Loss
  criterion = EmbedderLoss()

  # model
  embedder.train()
  recovery.train()

  # Optimizer
  # models_param = [embedder.parameters(), recovery.parameters()]
  # optimizer = torch.optim.Adam(params=itertools.chain(*models_param), lr=learning_rate)
  optimizer = torch.optim.Adam(
    [{'params': embedder.parameters()},
     {'params': recovery.parameters()}],
    lr=H_config["learning rate1"]
  )

  print('Start Embedding Network Training')

  num_epochs = H_config["num_epochs"]

  for epoch in range(num_epochs):

    training_loss = 0.0

    for i, inputs in enumerate(data_loader):

      X = inputs[0].to(CUDA_DEVICES)

      optimizer.zero_grad()

      H = embedder(X, None)
      outputs = recovery(H, None)

      loss_only, loss = criterion(outputs, X)
      loss.backward()
      optimizer.step()

      training_loss += loss_only.item() * X.size(0)

    training_loss = training_loss / len(data_set)

    if epoch % (np.round(num_epochs / 5))  == 0:
      print('epoch: '+ str(epoch) + '/' + str(num_epochs) + ', e_loss: ' + str(np.round(np.sqrt(training_loss),4)))

    tune.report(loss=training_loss)

  print('Finish Embedding Network Training')

# 2. Training only with supervised loss
def train_stage2(H_config, embedder, supervisor, generator):

  # Dataset
  data_set = SensorSignalDataset(root_dir=dataset_dir, transform=None)
  data_loader = DataLoader(dataset=data_set, batch_size=H_config["batch_size"], shuffle=False, num_workers=1)

  # Loss
  criterion = SupervisedLoss()

  # model
  embedder.train()
  supervisor.train()
  generator.train()

  # Optimizer
  # models_param = [generator.parameters(), supervisor.parameters()]
  # optimizer = torch.optim.Adam(params=itertools.chain(*models_param), lr=learning_rate)

  optimizer = torch.optim.Adam(
    [{'params': generator.parameters()},
     {'params': supervisor.parameters()}],
    lr=H_config["learning rate2"]
  )

  print('Start Training with Supervised Loss Only')

  num_epochs = H_config["num_epochs"]

  for epoch in range(num_epochs):

    training_loss = 0.0

    for i, inputs in enumerate(data_loader):

      X = inputs[0].to(CUDA_DEVICES)

      optimizer.zero_grad()

      H = embedder(X, None)
      H_hat_supervise = supervisor(H, None)

      loss = criterion(H_hat_supervise[:,:-1,:], H[:,1:,:])
      loss.backward()
      optimizer.step()

      training_loss += loss.item() * X.size(0)

    training_loss = training_loss / len(data_set)

    if epoch % (np.round(num_epochs / 5)) == 0:
      print('epoch: '+ str(epoch) + '/' + str(num_epochs) + ', s_loss: ' + str(np.round(np.sqrt(training_loss),4)))

    tune.report(loss=training_loss)
  print('Finish Training with Supervised Loss Only')

# 3. Joint Training
def train_stage3(H_config, embedder, recovery, generator, supervisor, discriminator):

  print('Start Joint Training')

  # Dataset
  data_set = SensorSignalDataset(root_dir=dataset_dir, transform=None)
  data_loader = DataLoader(dataset=data_set, batch_size=H_config["batch_size"], shuffle=False, num_workers=1)

  # generator loss
  Gloss_criterion = JointGloss()
  Eloss_criterion = JointEloss()
  Dloss_criterion = JointDloss()


  # model
  embedder.train()
  recovery.train()
  generator.train()
  supervisor.train()
  discriminator.train()

  # optimizer
  # models_paramG = [generator.parameters(), supervisor.parameters()]
  # optimizerG = torch.optim.Adam(params=itertools.chain(*models_paramG), lr=learning_rate)

  optimizerG = torch.optim.Adam(
    [{'params': generator.parameters()},
     {'params': supervisor.parameters()}],
    lr=H_config["learning rate3"]
  )

  # models_paramE = [embedder.parameters(), recovery.parameters()]
  # optimizerE = torch.optim.Adam(params=itertools.chain(*models_paramE), lr=learning_rate)

  optimizerE = torch.optim.Adam(
    [{'params': embedder.parameters()},
     {'params': recovery.parameters()}],
    lr=H_config["learning rate4"]
  )

  optimizerD = torch.optim.Adam(params=discriminator.parameters(), lr=H_config["learning rate5"])

  num_epochs = H_config["num_epochs"]

  for epoch in range(num_epochs):

    training_loss_G = 0.0
    training_loss_U = 0.0
    training_loss_S = 0.0
    training_loss_V = 0.0
    training_loss_E0 = 0.0
    training_loss_D = 0.0

    # Generator training (twice more than discriminator training)
    for _ in range(2):
      for inputs in data_loader:

        X = inputs[0].to(CUDA_DEVICES)

        optimizerG.zero_grad()
        optimizerE.zero_grad()

        # Train generator
        z_batch_size, z_seq_len, z_dim = X.shape
        Z = random_generator(z_batch_size, z_seq_len, z_dim)
        Z = Z.to(CUDA_DEVICES)

        E_hat = generator(Z, None)
        H_hat = supervisor(E_hat, None)
        Y_fake = discriminator(H_hat, None)
        Y_fake_e = discriminator(E_hat, None)
        H = embedder(X, None)
        X_tilde = recovery(H, None)
        H_hat_supervise = supervisor(H, None)
        X_hat = recovery(H_hat, None)

        lossG, loss_U, loss_S, loss_V = Gloss_criterion(Y_fake, Y_fake_e, H[:,1:,:], H_hat_supervise[:,:-1,:], X, X_hat)

        lossG.backward()
        optimizerG.step()


        training_loss_G += lossG.item() * X.size(0)
        training_loss_U += loss_U.item() * X.size(0)
        training_loss_S += loss_S.item() * X.size(0)
        training_loss_V += loss_V.item() * X.size(0)

        # Train embedder

        H = embedder(X, None)
        X_tilde = recovery(H, None)
        H_hat_supervise = supervisor(H, None)

        lossE, lossE_0 = Eloss_criterion(X_tilde, X, H[:,1:,:], H_hat_supervise[:,:-1,:])

        lossE.backward()
        optimizerE.step()

        training_loss_E0 += lossE_0.item() * X.size(0)


    # Discriminator training
    for i, inputs in enumerate(data_loader):

      X = inputs[0].to(CUDA_DEVICES)
      optimizerD.zero_grad()

      z_batch_size, z_seq_len, z_dim = X.shape
      Z = random_generator(z_batch_size, z_seq_len, z_dim)
      Z = Z.to(CUDA_DEVICES)

      E_hat = generator(Z, None)
      Y_fake_e = discriminator(E_hat, None)
      H_hat = supervisor(E_hat, None)
      Y_fake = discriminator(H_hat, None)

      H = embedder(X, None)
      Y_real = discriminator(H, None)

      lossD = Dloss_criterion(Y_real, Y_fake, Y_fake_e)

      # Train discriminator (only when the discriminator does not work well)
      if lossD > 0.15:
        lossD.backward()
        optimizerD.step()
        training_loss_D += lossD.item() * X.size(0)

    training_loss_G = 0.5 * (training_loss_G / len(data_set))
    training_loss_U = 0.5 * (training_loss_U / len(data_set))
    training_loss_S = 0.5 * (training_loss_S / len(data_set))
    training_loss_V = 0.5 * (training_loss_V / len(data_set))
    training_loss_E0 = 0.5 * (training_loss_E0 / len(data_set))
    training_loss_D = training_loss_D / len(data_set)

    tune.report(loss=((training_loss_G+training_loss_U+training_loss_S+training_loss_V+training_loss_E0+training_loss_D)/6.0))


    # Print multiple checkpoints
    if epoch % (np.round(num_epochs / 5)) == 0:
      print('step: '+ str(epoch) + '/' + str(num_epochs) +
            ', d_loss: ' + str(np.round(training_loss_D, 4)) +
            ', g_loss_u: ' + str(np.round(training_loss_U, 4)) +
            ', g_loss_s: ' + str(np.round(np.sqrt(training_loss_S), 4)) +
            ', g_loss_v: ' + str(np.round(training_loss_V, 4)) +
            ', e_loss_t0: ' + str(np.round(np.sqrt(training_loss_E0), 4)))

      epoch_embedder_name = str(epoch) + "_" + embedder_name
      epoch_recovery_name = str(epoch) + "_" + recovery_name
      epoch_generator_name = str(epoch) + "_" + generator_name
      epoch_supervisor_name = str(epoch) + "_" + supervisor_name
      epoch_discriminator_name = str(epoch) + "_" + discriminator_name

      # save model
      today = date.today()
      save_time = today.strftime("%d_%m_%Y")
      output_dir = config.get('train', 'model_path') + '/' + save_time + '/' + classification_dir + '/'
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
  
      torch.save(embedder, f'{output_dir+epoch_embedder_name}')
      torch.save(recovery, f'{output_dir+epoch_recovery_name}')
      torch.save(generator, f'{output_dir+epoch_generator_name}')
      torch.save(supervisor, f'{output_dir+epoch_supervisor_name}')
      torch.save(discriminator, f'{output_dir+epoch_discriminator_name}')

  print('Finish Joint Training')


def train(H_config):
  # models
  embedder = Embedder(
    module = module_name,
    time_stamp = seq_len,
    input_size = n_features,
    hidden_dim = H_config["hidden_size"],
    output_dim = H_config["hidden_size"],
    num_layers = H_config["num_layers"],
    activate_function=nn.Tanh()
  )

  recovery = Recovery(
    module = module_name,
    time_stamp = seq_len,
    input_size = H_config["hidden_size"],
    hidden_dim = H_config["hidden_size"],
    output_dim = n_features,
    num_layers = H_config["num_layers"],
    activate_function=nn.Tanh() 
  )

  generator = Generator(
    module = module_name,
    time_stamp = seq_len,
    input_size = n_features,
    hidden_dim = H_config["hidden_size"],
    output_dim = H_config["hidden_size"],
    num_layers = H_config["num_layers"],
    activate_function=nn.Tanh()
  )

  supervisor = Supervisor(
    module = module_name,
    time_stamp = seq_len,
    input_size = H_config["hidden_size"],
    hidden_dim = H_config["hidden_size"],
    output_dim = H_config["hidden_size"],
    # [Supervisor] num_layers must less(-1) than other component, embedder
    num_layers = H_config["num_layers"] - 1,
    activate_function=nn.Tanh()
  )

  discriminator = Discriminator(
    module = module_name,
    time_stamp = seq_len,
    input_size = H_config["hidden_size"],
    hidden_dim = H_config["hidden_size"],
    output_dim = 1,
    num_layers = H_config["num_layers"],
  )

  embedder = embedder.to(CUDA_DEVICES)
  recovery = recovery.to(CUDA_DEVICES)
  generator = generator.to(CUDA_DEVICES)
  supervisor = supervisor.to(CUDA_DEVICES)
  discriminator = discriminator.to(CUDA_DEVICES)

  train_stage1(H_config, embedder, recovery)
  train_stage2(H_config, embedder, supervisor, generator)
  train_stage3(H_config, embedder, recovery, generator, supervisor, discriminator)

  # save model
  today = date.today()
  save_time = today.strftime("%d_%m_%Y")
  output_dir = config.get('train', 'model_path') + '/' + save_time + '/' + config.get('train', 'classification_dir') + '/'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  torch.save(embedder, f'{output_dir+embedder_name}')
  torch.save(recovery, f'{output_dir+recovery_name}')
  torch.save(generator, f'{output_dir+generator_name}')
  torch.save(supervisor, f'{output_dir+supervisor_name}')
  torch.save(discriminator, f'{output_dir+discriminator_name}')


if __name__ == '__main__':

  # Parameters

  Config_opt = {
    "num_epochs": tune.choice([20, 40, 100, 150, 200, 250, 500, 1000]),
    "batch_size": tune.choice([8, 16, 32, 64, 128]),
    "hidden_size": tune.choice([50, 100 , 150, 200, 300, 400, 500]),
    "num_layers": tune.choice([5, 10, 15, 20, 30]),
    "learning rate1": tune.loguniform(1e-4, 1e-1),
    "learning rate2": tune.loguniform(1e-4, 1e-1),
    "learning rate3": tune.loguniform(1e-4, 1e-1),
    "learning rate4": tune.loguniform(1e-4, 1e-1),
    "learning rate5": tune.loguniform(1e-4, 1e-1),
  }
  scheduler = ASHAScheduler(
          metric="loss",
          mode="min",
          max_t=500,
          grace_period=1,
          reduction_factor=2)
  reporter = CLIReporter(
          # parameter_columns=["l1", "l2", "lr", "batch_size"],
          metric_columns=["loss"])

  result = tune.run(
          train,
          resources_per_trial={"cpu": 1, "gpu": 1},
          config=Config_opt,
          num_samples=10000,
          scheduler=scheduler,
          progress_reporter=reporter)

  best_trial = result.get_best_trial("loss", "min", "last")
  print("Best trial config: {}".format(best_trial.config))
  print("Best trial final loss: {}".format(best_trial.last_result["loss"]))
 
