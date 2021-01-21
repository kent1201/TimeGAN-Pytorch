import math
import torch
import torch.nn as nn

class SupervisedLoss(nn.Module):

  def __init__(self):
    super(SupervisedLoss, self).__init__()
    self.MSELoss = nn.MSELoss()

  def forward(self, outputs, targets):

    loss = self.MSELoss(outputs, targets)
    
    return loss 


if __name__ == '__main__':
  
  outputs = torch.randn(32, 82, 24)
  targets = torch.randn(32, 82, 24)
  
  criterion = SupervisedLoss()
  loss = criterion(outputs, targets)
  print(loss)
  



