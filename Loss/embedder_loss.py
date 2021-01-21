import math
import torch
import torch.nn as nn
from Loss.soft_dtw_cuda import SoftDTW

class EmbedderLoss(nn.Module):

  def __init__(self):
    super(EmbedderLoss, self).__init__()
    self.MSELoss = nn.MSELoss()

  def forward(self, outputs, targets):

    loss_only = self.MSELoss(outputs, targets)
    loss = torch.mul(10.0, torch.sqrt(torch.add(loss_only, 1e-10)))
    
    return loss_only, loss 


if __name__ == '__main__':
  
  outputs = torch.randn(32, 82, 24)
  targets = torch.randn(32, 82, 24)
  
  criterion = EmbedderLoss()
  loss_only, loss = criterion(outputs, targets)
  print(loss)
  



