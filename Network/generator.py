import torch.nn as nn
import torch
import torch.nn.functional as F
from Network.tcn import TemporalConvNet

class Generator(nn.Module):
  def __init__(self, module='gru', time_stamp=82, input_size=27, hidden_dim=100, output_dim=100, num_layers=10, activate_function=nn.Tanh()):
    super(Generator, self).__init__()

    self.module = module
    self.input_size = input_size
    self.time_stamp = time_stamp
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.hidden_dim_layers = []
    self.output_dim = output_dim

    if self.module == 'gru':
      self.r_cell = nn.GRU(
        input_size = self.input_size,
        hidden_size = self.hidden_dim,
        num_layers = self.num_layers,
        batch_first = True
      )
    elif self.module == 'lstm':
      self.r_cell = nn.LSTM(
        input_size = self.input_size,
        hidden_size = self.hidden_dim,
        num_layers = self.num_layers,
        batch_first = True
      )
    elif self.module == 'bi-lstm':
      self.r_cell = nn.LSTM(
        input_size = self.input_size,
        hidden_size = self.hidden_dim,
        num_layers = self.num_layers,
        batch_first = True,
        bidirectional = True
      )

    elif self.module == 'tcn':
      
      for i in range(num_layers):
        self.hidden_dim_layers.append(self.hidden_dim)
      
      self.r_cell = TemporalConvNet( num_inputs=self.input_size, 
        num_channels=self.hidden_dim_layers, 
        kernel_size=2,
        dropout=0.2
      )

    if self.module == 'bi-lstm':
      self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
    else:
      self.fc = nn.Linear(self.hidden_dim, self.output_dim)
    
    self.activate = activate_function

  def forward(self, X, H):
    if self.module == 'tcn':
      # Input X shape: (batch_size, seq_len, input_dim)
      X = torch.transpose(X, 1, 2)
      # Input X shape: (batch_size, input_dim, seq_len)
      output = self.r_cell(X)
      # H shape: (batch_size, input_dim, seq_len)
      # H_transpose: (batch_size, seq_len, input_dim)
      output = torch.transpose(output, 1, 2)
    else:
      # Inpu t X shape: (seq_len, batch_size, input_dim)
      output, _ = self.r_cell(X, H)
      # Outputs shape: (seq_len, batch_size, input_dim)

    H = self.fc(self.activate(output))
    
    return H

# gpu-used
CUDA_DEVICES = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
  model = Generator(
    module = 'tcn',
    time_stamp = 82,
    input_size = 27,
    hidden_dim = 100,
    output_dim = 100,
    num_layers = 10,
    activate_function=nn.Tanh()
  )
  model = model.to(CUDA_DEVICES)
  model.train()
  inputs = torch.randn(32, 82, 27)
  inputs = inputs.to(CUDA_DEVICES)
  outputs = model(inputs, None)
  print("[generator.py] model: {}".format(model))
  print("[generator.py] inputs: {}".format(inputs.shape))
  print("[generator.py] outputs: {}".format(outputs.shape))


if __name__ == '__main__':
  test()






    
