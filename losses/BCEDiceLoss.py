from torch import nn
import torch
import numpy as np


class DiceLoss(nn.Module):
    """Calculate dice loss."""

    def __init__(
        self, apply_softmax: bool = True, smooth: float = 1.0, eps: float = 1e-7  # -7
    ):
        # eps is used to avoid division by zero
        # for float16 use 1e-7
        # for float32 use 1e-8~1e-9
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.smooth = smooth
        self.apply_softmax = apply_softmax

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.apply_softmax:
            predict = torch.sigmoid(predict)

        num = target.size(0)
        probability = predict.view(num, -1)
        target = target.view(num, -1)
        assert probability.shape == target.shape

        intersection = 2.0 * (probability * target).sum()
        union = probability.sum() + target.sum()
        dice_score = (intersection + self.smooth) / (union + self.eps + self.smooth)
        return 1.0 - dice_score


class BinaryDiceLoss(DiceLoss):
    """
    Compute Dice Loss for binary class tasks (1 class only).
    Except target to be a binary map with 0 and 1 values.
    """

    def __init__(
        self, apply_sigmoid: bool = True, smooth: float = 1.0, eps: float = 1e-7
    ):
        """
        :param apply_sigmoid: Whether to apply sigmoid to the predictions.
        :param smooth: laplace smoothing, also known as additive smoothing. The larger smooth value is, closer the dice
            coefficient is to 1, which can be used as a regularization effect.
            As mentioned in: https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895
        :param eps: epsilon value to avoid inf.
        """
        super().__init__(
            apply_softmax=False,
            smooth=smooth,
            eps=eps,
        )
        self.apply_sigmoid = apply_sigmoid

    def forward(self, predict: torch.tensor, target: torch.tensor) -> torch.tensor:
        if self.apply_sigmoid:
            predict = torch.sigmoid(predict)
        return super().forward(predict=predict, target=target)


class BCEDiceLoss(nn.Module):
    """
    alias: DetailLoss
    STDC DetailLoss applied on  details features from higher resolution and ground-truth details map.
    Loss combination of BCE loss and BinaryDice loss
    """

    def __init__(self, weights: list = [1.0, 1.0]):
        """
        :param weights: weight to apply for each part of the loss contributions, [BCE, Dice] respectively.
        """
        super().__init__()
        assert (
            len(weights) == 2
        ), f"Only 2 weight elements are required for BCE-Dice loss combo, found: {len(weights)}"
        self.weights = weights
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.dice_loss = BinaryDiceLoss(apply_sigmoid=True)

    def forward(self, detail_out: torch.Tensor, detail_target: torch.Tensor):
        """
        :param detail_out: predicted detail map.
        :param detail_target: ground-truth detail loss, output of DetailAggregateModule.
        """
        # assert (
        #     torch.isnan(detail_out).any() == False
        #     and torch.isnan(detail_target).any() == False
        # ), "detail_out or detail_target has nan"

        bce_loss = self.bce_with_logits(detail_out, detail_target)
        dice_loss = self.dice_loss(detail_out, detail_target)
        return self.weights[0] * bce_loss + self.weights[1] * dice_loss


def dice_coef_metric(
    probabilities: torch.Tensor,
    truth: torch.Tensor,
    treshold: float = 0.5,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Calculate Dice score for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert predictions.shape == truth.shape
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


if __name__ == "__main__":
    import torch

    y_pred = torch.ones(1, 1, 256, 256, 50)
    y_true = torch.ones(1, 1, 256, 256, 50)
    dice = DiceLoss()
    loss = dice(y_pred, y_true)
    print(loss)

    criteria = BCEDiceLoss()
    y_pred = torch.randn(1, 1, 256, 256, 50)
    y_true = torch.randn(1, 1, 256, 256, 50)
    loss = criteria(y_pred, y_true)
    print(loss)
