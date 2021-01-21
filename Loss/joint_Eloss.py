import math
import torch
import torch.nn as nn
from Loss.soft_dtw_cuda import SoftDTW

class JointEloss(nn.Module):

  def __init__(self):
    super(JointEloss, self).__init__()
    self.MSELoss = nn.MSELoss()

  def forward(self, X_tilde, X, H, H_hat_supervise):

    loss_T0 = self.MSELoss(X_tilde, X)
    loss_0 = torch.mul(torch.sqrt(torch.add(loss_T0, 1e-10)), 10)
    G_loss_S = torch.mul(self.MSELoss(H_hat_supervise, H), 0.1)
    loss = torch.add(loss_0, G_loss_S)

    return loss , loss_T0


if __name__ == '__main__':

  X_tilde = torch.randn(32, 82, 34)
  X = torch.randn(32, 82, 34)
  H = torch.randn(32, 82, 24)
  H_hat_supervise = torch.randn(32, 82, 24)

  criterion = JointEloss()
  loss = criterion(X_tilde, X, H, H_hat_supervise)
  print(loss)