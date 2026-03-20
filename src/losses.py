import torch.nn as nn
import torch
import numpy as np

class MaskedMSELoss(nn.Module):
    # masked MSE loss for nans
    def __init__(self, R=None, B=None):
        super().__init__()
        # response matrix R
        self.R = R 
        # basis matrix B
        self.B = B

    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        diff = pred[mask] - target[mask]
        return torch.mean(diff ** 2)

class JointResynthLoss(nn.Module):
    def __init__(self, R, transformed=False):
        super().__init__()
        self.R = R.T  # [n_bins, C] -> [C, n_bins]
        self.transformed = transformed

    def forward(self, pred, target, aia_obs):
        # unbatched fix
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            aia_obs = aia_obs.unsqueeze(0)
        
        # dem loss
        mask = ~torch.isnan(target)
        dem_diff = pred[mask] - target[mask]
        dem_loss = torch.mean(dem_diff ** 2)

        # resynthesis
        if self.transformed:
            pred = pred ** 2
        
        B, n_bins, H, W = pred.shape
        pred_reshaped = pred.permute(0, 2, 3, 1)  # [B, H, W, n_bins]
        synth = torch.matmul(pred_reshaped, self.R.to(pred.device))  # [B, H, W, C]
        synth = synth.permute(0, 3, 1, 2)  # [B, C, H, W]

        if self.transformed:
            synth = torch.sqrt(torch.clamp(synth, min=0))

        # trick, scale synth to match the dem scale
        # so we have both mse losses in the same scale
        with torch.no_grad():
            scale = pred.mean() / (synth.mean() + 1e-8)
        synth = synth * scale
        aia_obs = aia_obs * scale
        
        # aia loss
        aia_mask = ~torch.isnan(aia_obs)
        aia_diff = synth[aia_mask] - aia_obs[aia_mask]
        aia_loss = torch.mean(aia_diff ** 2)

        return dem_loss + aia_loss

class ClassificationLoss(nn.Module):
    def __init__(self, bins, ignore_index=-1, R=None, B=None):
        super().__init__()
        self.bins = bins
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        # response matrix R
        self.R = R 
        # basis matrix B
        self.B = B
    
    def forward(self, cls_pred, bin_targets):
        # cls_pred: [B, 26, 64, H, W] - logits
        # bin_targets: [B, 26, H, W] - target bin indices
        
        B, n_temps, n_bins, H, W = cls_pred.shape
        
        # reshape for cross-entropy: [B*26*H*W, 64] and [B*26*H*W]
        cls_pred_flat = cls_pred.permute(0, 1, 3, 4, 2).reshape(-1, n_bins)  
        bin_targets_flat = bin_targets.reshape(-1)
        
        # mask out invalid targets
        valid_mask = (bin_targets_flat >= 0) & (bin_targets_flat < n_bins)
        if valid_mask.sum() > 0:
            return self.ce_loss(cls_pred_flat[valid_mask], bin_targets_flat[valid_mask])
        else:
            return cls_pred_flat.sum() * 0.0  # preserve grad, avoid tensor creation

class RegressionClassificationLoss(nn.Module):
    def __init__(self, bins, alpha=0.5, R=None, B=None):
        super().__init__()
        self.bins = bins
        self.alpha = alpha  # weight between regression and classification
        self.mse_loss = MaskedMSELoss(R=R, B=B)
        self.cls_loss = ClassificationLoss(bins, R=R, B=B)
        self.R = R 
        self.B = B
    
    def forward(self, reg_pred, cls_pred, dem_target, bin_targets):
        reg_loss = self.mse_loss(reg_pred, dem_target)
        cls_loss = self.cls_loss(cls_pred, bin_targets)
        return self.alpha * reg_loss + (1 - self.alpha) * cls_loss
    
def barrier_loss_batch(x, D, I_obs, lb, ub, a_l2=0, a_l1=1.0, mu=1.0, alpha=0, mu_pos=0):
    # x: [B, n_basis]
    # D: [n_obs, n_temps]
    # I_obs: [B, n_obs]
    # lb, ub: [B, n_obs] lower and upper bounds for the observed data

    # a_l2: regularization strength for L2
    # a_l1: regularization strength for L1
    # mu: barrier strength
    # alpha: fit term weight

    # L2 regularization
    l2_term = 0.5 * a_l2 * torch.sum(x**2, dim=1)  # [B]

    # L1 regularization
    l1_term = a_l1 * torch.sum(torch.abs(x), dim=1)  # [B]

    Dx = torch.matmul(x, D.T)  # [B, n_obs]

    # barrier for x >= 0
    barrier_x = mu_pos * torch.sum(torch.relu(-x), dim=1)  # [B]
    
    # barrier for Dx >= lb
    barrier_lb = mu * torch.sum(torch.relu(lb - Dx)**2, dim=1)  # [B]
    #barrier_lb = mu * torch.sum(nn.functional.softplus(lb - Dx), dim=1)  # [B]

    # barrier for Dx <= ub
    barrier_ub = mu * torch.sum(torch.relu(Dx - ub)**2, dim=1)  # [B]
    #barrier_ub = mu * torch.sum(nn.functional.softplus(Dx - ub), dim=1)  # [B]

    # fit term
    fit = alpha * torch.sum((Dx - I_obs)**2, dim=1)  # [B]

    # final loss: shape [B], return mean for scalar loss
    total_loss = l2_term + l1_term + barrier_x + barrier_lb + barrier_ub + fit
    return total_loss.mean()

class BarrierLoss(nn.Module):
    def __init__(self, D, R, B, args=None):
        super().__init__()
        self.D = D
        self.R = R
        self.B = B
        self.a_l2 = args.alpha_l2
        self.a_l1 = args.alpha_l1
        self.mu = args.mu
        self.alpha = args.alpha_fit

    def forward(self, x, aia_obs, lb, ub):
        # lb: lower bound, shape [B, n_obs]
        # ub: upper bound, shape [B, n_obs]
        return barrier_loss_batch(x, self.D, aia_obs, lb, ub,
                                  self.a_l2, self.a_l1, self.mu, self.alpha)