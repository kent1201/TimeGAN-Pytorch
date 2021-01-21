import math
import torch
import torch.nn as nn
from Loss.soft_dtw_cuda import SoftDTW

class JointGloss(nn.Module):

  def __init__(self):
    super(JointGloss, self).__init__()
    # Adversarial loss
    # equivalent as sigmoid_cross_entropy_with_logits
    self.G_loss_U = nn.BCEWithLogitsLoss()
    self.gamma = 1
    # Supervised loss
    self.G_loss_S = nn.MSELoss()
    # Two Momments (計算合成 data 與原始 data 的 loss)
     

  def forward(self, Y_fake, Y_fake_e, H, H_hat_supervise, X, X_hat):

    """
      Y_fake, Y_fake_e: [batch_size, seq_len, 1]
      H, H_hat_supervise: [batch_size, seq_len-1, n_features(hidden)]
      X, X_hat: [batch_size, seq_len, n_features]
    """
    loss_V1 = torch.mean(torch.abs(torch.sub(torch.sqrt(torch.add(torch.var(X_hat, dim=0, keepdim=True, unbiased=True), 1e-7)), torch.sqrt(torch.add(torch.var(X, dim=0, keepdim=True, unbiased=True), 1e-7)))))
    loss_V2 = torch.mean(torch.abs(torch.sub(torch.mean(X_hat, dim=0, keepdim=True), torch.mean(X, dim=0, keepdim=True))))
    loss_V = loss_V1.add(loss_V2)
    loss_U = self.G_loss_U(Y_fake, torch.ones_like(Y_fake)) 
    loss_U_e = self.G_loss_U(Y_fake_e, torch.ones_like(Y_fake_e))
    loss_U = loss_U.add(torch.mul(self.gamma, loss_U_e))
    loss_S = torch.mul(torch.sqrt(torch.add(self.G_loss_S(H_hat_supervise, H), 1e-7)), 100)
    loss = loss_U.add(loss_S).add(torch.mul(loss_V, 100))
    
    return loss, loss_U, loss_S, loss_V


if __name__ == '__main__':
  
  Y_fake = torch.randn(32, 82, 1)
  Y_fake_e = torch.randn(32, 82, 1)
  H = torch.randn(32, 81, 24)
  H_hat_supervise = torch.randn(32, 81, 24)
  X = torch.randn(32, 82, 34)
  X_hat = torch.randn(32, 82, 34)
  
  criterion = JointGloss()
  loss = criterion(Y_fake, Y_fake_e, H, H_hat_supervise, X, X_hat)
  print(loss)
  