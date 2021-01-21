import torch.nn as nn
import torch
import torch.nn.functional as F

class Simple_Predictor(nn.Module):
  def __init__(self, time_stamp=81, input_size=27, hidden_dim=136, output_dim=27, num_layers=2):
    super(Simple_Predictor, self).__init__()
    self.input_size = input_size
    self.time_stamp = time_stamp
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.output_dim = output_dim
    self.r_cell = nn.GRU(
        input_size = self.input_size,
        hidden_size = self.hidden_dim,
        num_layers = self.num_layers,
        batch_first = True
    )
    self.activate1 = nn.Tanh()
    self.activate2 = nn.Sigmoid()
    self.fc = nn.Linear(self.hidden_dim, self.output_dim)

  def forward(self, X, H):
    # Input X shape: (seq_len, batch_size, input_dim)
    outputs1, _ = self.r_cell(X, H)
    outputs2 = self.activate1(outputs1)
    # Outputs shape: (seq_len, batch_size, input_dim)
    H = self.activate2(self.fc(outputs2[:, X.shape[1]-1, :]))
    return H


# gpu-used
CUDA_DEVICES = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test():
  model = Simple_Predictor(
    time_stamp = 81,
    input_size = 33,
    hidden_dim = 17,
    output_dim = 1,
    num_layers = 2
  )
  model = model.to(CUDA_DEVICES)
  model.train()
  inputs = torch.randn(32, 82, 33)
  inputs = inputs.to(CUDA_DEVICES)
  outputs = model(inputs, None)
  print("[discriminate.py main] model: {}".format(model))
  print("[discriminate.py main] inputs: {}".format(inputs.shape))
  print("[discriminate.py main] outputs: {}".format(outputs[0].data))


if __name__ == '__main__':
  test()







