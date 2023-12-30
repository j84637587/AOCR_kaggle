import torch
import numpy as np
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryAUROC,
)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix="", log_path="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_path = log_path
        self.logger = logger

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.logger:
            self.logger.info("\t".join(entries))
        else:
            print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.log = []

    def update(self, val, n=1):
        self.log.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmt_str = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmt_str.format(**self.__dict__)


def dice_coef_metric(
    probabilities: torch.Tensor,
    truth: torch.Tensor,
    treshold: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Calculate Dice score for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.

    .. math::
        F_{1} = 2\frac{\text{precision} * \text{recall}}{(\text{precision}) + \text{recall}}
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


def jaccard_coef_metric(
    probabilities: torch.Tensor,
    truth: torch.Tensor,
    treshold: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Calculate Jaccard index for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: jaccard score aka iou."
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert predictions.shape == truth.shape

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


class Meter:
    def __init__(self, treshold: float = 0.5):
        self.threshold: float = treshold
        self.dice_scores: list = []
        self.iou_scores: list = []
        self.f1_scores: list = []
        self.auroc_score: list = []

        self.m_f1 = BinaryF1Score().cuda()
        self.m_auroc = BinaryAUROC().cuda()

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        dice = dice_coef_metric(probs, targets, self.threshold)
        iou = jaccard_coef_metric(probs, targets, self.threshold)
        f1 = self.m_f1(probs, targets).item()
        auroc = self.m_auroc(probs, targets).item()

        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.f1_scores.append(f1)
        self.auroc_score.append(auroc)

    def get_last_metrics(self) -> np.ndarray:
        dice = self.dice_scores[-1]
        iou = self.iou_scores[-1]
        f1 = self.f1_scores[-1]
        auroc = self.auroc_score[-1]
        return dice, iou, f1, auroc

    def get_metrics(self) -> np.ndarray:
        dice = np.mean(self.dice_scores)
        iou = np.mean(self.iou_scores)
        f1 = np.mean(self.f1_scores)
        auroc = np.mean(self.auroc_score)
        return dice, iou, f1, auroc


def build_train_progress(data_size):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    return ProgressMeter(
        data_size,
        batch_time,
        data_time,
        losses,
        top1,
        top5,
        prefix="Epoch: [{}]".format(1),
    )


def build_valid_progress(data_size):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    return ProgressMeter(data_size, batch_time, losses, top1, top5, prefix="Test: ")
