import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)"""

    def __init__(
        self, loss_fcn: nn.BCEWithLogitsLoss, gamma: float = 1.5, alpha: float = 0.25
    ):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FocalLoss to each element

    def forward(self, pred: torch.tensor, true: torch.tensor) -> torch.tensor:
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


if __name__ == "__main__":
    criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    y_pred = torch.randn(1, 1, 256, 256, 50)
    y_true = torch.randn(1, 1, 256, 256, 50)
    loss = criteria(y_pred, y_true)
    print(loss)
