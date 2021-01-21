import torch.nn as nn
import torch
import torch.nn.functional as F
from Network.TCN.tcn import TemporalConvNet

class Embedder(nn.Module):
  def __init__(self, time_stamp=82, input_size=34, hidden_dim=12, output_dim=24, num_layers=10, activate_function=nn.Tanh()):
    super(Embedder, self).__init__()
    self.input_size = input_size
    self.time_stamp = time_stamp
    self.hidden_dim = hidden_dim
    self.hidden_dim_layers = []
    self.output_dim = output_dim

    for i in range(num_layers):
      self.hidden_dim_layers.append(hidden_dim)

    self.tcn = TemporalConvNet( num_inputs=self.input_size, 
        num_channels=self.hidden_dim_layers, 
        kernel_size=2,
        dropout=0.2
    )
    self.fcc = nn.Linear(self.hidden_dim, self.output_dim)
    self.activate_function = activate_function

  def forward(self, X):
    # Input X shape: (batch_size, seq_len, input_dim)
    X = torch.transpose(X, 1, 2)
     # Input X shape: (batch_size, input_dim, seq_len)
    H = self.tcn(X)
    # H shape: (batch_size, input_dim, seq_len)
    # H_transpose: (batch_size, seq_len, input_dim)
    H_transpose = torch.transpose(H, 1, 2)
    output = self.fcc(self.activate_function(H_transpose))
    return output


# gpu-used
CUDA_DEVICES = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
  model = Embedder(
    time_stamp = 82,
    input_size = 34,
    hidden_dim = 12,
    output_dim = 24,
    num_layers = 3,
    activate_function=nn.Tanh()
  )
  model = model.to(CUDA_DEVICES)
  model.train()
  inputs = torch.randn(32, 82, 34)
  inputs = inputs.to(CUDA_DEVICES)
  outputs = model(inputs)
  print("[embedder.py] model: {}".format(model))
  print("[embedder.py] inputs: {}".format(inputs.shape))
  print("[embedder.py] outputs: {}".format(outputs.shape))


if __name__ == '__main__':
  test()






    
