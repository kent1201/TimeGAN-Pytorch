import math
import torch
import torch.nn as nn
from Loss.soft_dtw_cuda import SoftDTW

class JointDloss(nn.Module):

  def __init__(self):
    super(JointDloss, self).__init__()
    self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    self.gamma = 1

  def forward(self, Y_real, Y_fake, Y_fake_e):

    loss_real = self.BCEWithLogitsLoss(Y_real, torch.ones_like(Y_real))
    loss_fake = self.BCEWithLogitsLoss(Y_fake, torch.zeros_like(Y_fake))
    loss_fake_e = self.BCEWithLogitsLoss(Y_fake_e, torch.zeros_like(Y_fake_e))
    loss = loss_real.add(loss_fake).add(torch.mul(loss_fake_e, self.gamma))
    
    return loss


if __name__ == '__main__':
  
  Y_real = torch.randn(32, 82, 1)
  Y_fake = torch.randn(32, 82, 1)
  Y_fake_e = torch.randn(32, 82, 1)
  
  criterion = JointDloss()
  loss = criterion(Y_real, Y_fake, Y_fake_e)
  print(loss)