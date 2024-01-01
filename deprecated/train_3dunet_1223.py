import argparse
import os
import random
import shutil
import time
import warnings
import json
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC,
)

import matplotlib.pyplot as plt

from losses.BCEDiceLoss import BCEDiceLoss, dice_coef_metric
from models.UNet3d import UNet3d
from datasets.mask_dataset import build_dataloader
from progressmeter import ProgressMeter, AverageMeter

parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=4,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    args.log_path, logger = create_logger()

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    logger.info("=> args" + str(args))
    logger.info("=> creating model CNN3D")
    model = UNet3d(in_channels=1, n_classes=1, n_channels=24)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    # criterion = nn.BCEWithLogitsLoss().cuda()
    criterion = BCEDiceLoss()

    # optimizer = torch.optim.SGD(
    #     model.parameters(),F
    #     args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay,
    # )

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        # weight_decay=args.weight_decay,
    )
    # scheduler = MultiStepLR(
    #     optimizer, milestones=[10], gamma=0.5, last_epoch=-1, verbose=True
    # )
    # scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, verbose=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.distributed:
        pass
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    csv_file_path = "data/TrainValid_split.csv"
    root_directory = "data/preprocess/232x176x50_m"
    # csv_ground_truth = "data/TrainValid_ground_truth.csv"
    # data\preprocess\232x176x50_m\1_Train,Valid_Image\Zx85587FA5B4B2A9CCF771CA89EF3F2F3D8CB02CD05A59A8CE.pt
    logger.info(f"=> root_directory {root_directory}")
    logger.info("=> loading training data")
    # transform = transforms.Compose([permute_and_expand_with_zeros])
    train_loader = build_dataloader(
        csv_file_path,
        root_directory,
        batch_size=args.batch_size,
        num_workers=args.workers,
        phase="Train",
    )

    logger.info("=> loading validating data")
    val_loader = build_dataloader(
        csv_file_path,
        root_directory,
        batch_size=args.batch_size,
        num_workers=args.workers,
        phase="Valid",
    )

    if args.evaluate:
        logger.info("=> evaluate")
        (
            acc,
            f1,
            precision,
            recall,
            loss,
            total_tp,
            total_fp,
            total_tn,
            total_fn,
        ) = validate(val_loader, model, criterion, None, logger, args)
        with open("res.txt", "w") as f:
            logger.info(f"loss: {loss}", file=f)
            logger.info(f"acc: {acc}", file=f)
            logger.info(f"precision: {precision}", file=f)
            logger.info(f"recall: {recall}", file=f)
            logger.info(f"f1: {f1}", file=f)
            logger.info(f"TP: {total_tp}", file=f)
            logger.info(f"FP: {total_fp}", file=f)
            logger.info(f"TN: {total_tn}", file=f)
            logger.info(f"FN: {total_fn}", file=f)
        return

    logger.info("=> start training")

    writer = SummaryWriter(args.log_path, flush_secs=5)
    args.writer = writer

    # init history

    history = {
        "train": {"loss": [], "acc": [], "dice_loss": []},
        "val": {
            "loss": [],
            "acc": [],
            "dice_loss": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "TP": [],
            "FP": [],
            "TN": [],
            "FN": [],
        },
    }

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_acc, train_loss, train_dice_loss = train(
            train_loader, model, criterion, optimizer, epoch, logger, args
        )

        # evaluate on validation set
        (
            val_acc,
            val_f1,
            val_precision,
            val_recall,
            val_loss,
            val_dice_loss,
            total_tp,
            total_fp,
            total_tn,
            total_fn,
        ) = validate(val_loader, model, criterion, epoch, logger, args)

        history["train"]["loss"].append(train_loss)
        history["train"]["dice_loss"].append(train_dice_loss)
        history["train"]["acc"].append(train_acc)
        history["val"]["loss"].append(val_loss)
        history["val"]["dice_loss"].append(val_dice_loss)
        history["val"]["acc"].append(val_acc)
        history["val"]["f1"].append(val_f1)
        history["val"]["precision"].append(val_precision)
        history["val"]["recall"].append(val_recall)
        history["val"]["TP"].append(total_tp)
        history["val"]["FP"].append(total_fp)
        history["val"]["TN"].append(total_tn)
        history["val"]["FN"].append(total_fn)

        # remember best acc@1 and save checkpoint
        is_best = val_f1 > best_acc1
        best_acc1 = max(val_f1, best_acc1)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": "Unet3d",
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                args.log_path,
            )
        plot_history(history, args.log_path, (epoch == args.epochs - 1))

        # scheduler.step()

    final_loss = history["val"]["loss"][-1]
    final_acc = history["val"]["acc"][-1]
    logger.info(f"Eval loss: {final_loss} - Eval acc: {final_acc}")
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, logger, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    dice_losses = AverageMeter("Dice Loss", ":.4e")
    acc = AverageMeter("Acc", ":.2f")
    progress = ProgressMeter(
        len(train_loader),
        batch_time,
        data_time,
        losses,
        dice_losses,
        acc,
        prefix="Epoch: [{}]".format(epoch),
        log_path=args.log_path,
        logger=logger,
    )

    m_dice = dice_coef_metric

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    for i, (id, images, target) in enumerate(train_loader):
        # logger.info(f"{images.shape} {target.shape}")

        # measure data loading time
        data_time.update(time.time() - end)
        images = torch.unsqueeze(images, 1)  # expand
        target = torch.unsqueeze(target, 1)  # expand

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)  # shape: BxTxCxHxW
        batch_loss = criterion(output, target)
        batch_acc = accuracy_binary(output, target)
        acc.update(batch_acc)
        losses.update(batch_loss.item(), images.size(0))
        dice_losses.update(m_dice(output.detach().cpu(), target.detach().cpu()))

        # compute gradient and do SGD step
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        args.writer.add_scalar("train_loss", batch_loss, epoch)
        args.writer.add_scalar("train_acc", batch_acc, epoch)

        if i % args.print_freq == 0:
            progress.print(i)

    return acc.avg, losses.avg, dice_losses.avg


def validate(val_loader, model, criterion, epoch, logger, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("BCE Loss", ":.4e")
    dice_losses = AverageMeter("Dice Loss", ":.4e")
    acc = AverageMeter("Acc", ":.2f")
    f1_score = AverageMeter("f1-score", ":.3f")
    precision = AverageMeter("precision", ":.3f")
    recall = AverageMeter("recall", ":.3f")
    auroc = AverageMeter("AUROC", ":.3f")

    progress = ProgressMeter(
        len(val_loader),
        batch_time,
        losses,
        dice_losses,
        acc,
        f1_score,
        precision,
        recall,
        auroc,
        prefix="Test: ",
        log_path=args.log_path,
        logger=logger,
    )

    m_f1 = BinaryF1Score().cuda()
    m_precision = BinaryPrecision().cuda()
    m_recall = BinaryRecall().cuda()
    m_auroc = BinaryAUROC().cuda()
    m_dice = dice_coef_metric

    # switch to evaluate mode
    model.eval()
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

    with torch.no_grad():
        end = time.time()
        for i, (id, images, target) in enumerate(val_loader):
            images = torch.unsqueeze(images, 1)  # expand
            target = torch.unsqueeze(target, 1)  # expand

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            batch_loss = criterion(output, target)

            # measure accuracy and record loss
            batch_acc = accuracy_binary(output, target)
            acc.update(batch_acc)

            # metrics
            f1_score.update(m_f1(output, target).item())
            precision.update(m_precision(output, target).item())
            recall.update(m_recall(output, target).item())
            auroc.update(m_auroc(output, target).item())
            dice_losses.update(m_dice(output.detach().cpu(), target.detach().cpu()))
            losses.update(batch_loss.item(), images.size(0))

            TP, FP, TN, FN = calculate_metrics(output, target)
            total_tp += TP
            total_fp += FP
            total_tn += TN
            total_fn += FN

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if epoch is not None:
                args.writer.add_scalar("val_loss", batch_loss, epoch)
                args.writer.add_scalar("val_acc", batch_acc, epoch)

            if i % args.print_freq == 0:
                progress.print(i)

    return (
        acc.avg,
        f1_score.avg,
        precision.avg,
        recall.avg,
        losses.avg,
        dice_losses.avg,
        total_tp,
        total_fp,
        total_tn,
        total_fn,
    )


def save_checkpoint(state, is_best, log_path):
    filename = os.path.join(log_path, "checkpoint.pth.tar")

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename,
            os.path.join(log_path, "model_best.pth.tar"),
        )


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy_binary(output, target):
    """Computes the binary accuracy"""
    with torch.no_grad():
        pred = torch.round(output)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        # logger.info(pred)

        correct = target.eq(pred.view_as(target)).sum().item()
        total = target.size(0) * target.size(1) * target.size(2) * target.size(3)
        acc = correct / total
        return acc


def plot_history(history, log_path, show=False):
    f, ax = plt.subplots(2, 3, figsize=(15, 5))
    f.subplots_adjust(hspace=0.5)  # Add gap between ax

    for i, name in enumerate(["loss", "acc", "dice_loss"]):
        ax[0][i].set_title(name)
        ax[0][i].plot(history["train"][name], label=f"train {name}")
        ax[0][i].plot(history["val"][name], label=f"val {name}")
        ax[0][i].set_xlabel("Epoch")
        ax[0][i].set_ylabel(name.capitalize())
        ax[0][i].legend()
        ax[0][i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax[0][2].axis("off")

    for i, name in enumerate(["f1", "precision", "recall"]):
        ax[1][i].set_title(name)
        ax[1][i].plot(history["val"][name], label=f"val {name}")
        ax[1][i].set_xlabel("Epoch")
        ax[1][i].set_ylabel(name.capitalize())
        ax[1][i].legend()
        ax[1][i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.savefig(os.path.join(log_path, "history.png"))
    with open(os.path.join(log_path, "history.json"), "w") as f:
        json.dump(history, f)
    if show:
        plt.show()
    else:
        plt.close()


def create_logger():
    i = 0
    while True:
        try:
            log_path = f"logs/{time.strftime('%Y_%m_%d')}_{i}"
            os.mkdir(log_path)
            break
        except FileExistsError:
            i += 1
            pass
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_path, "run.log")),  # Log to a file
            logging.StreamHandler(),  # Log to the console
        ],
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(f"=> Log path: {log_path}")
    return log_path, logger


def calculate_metrics(predicted, target, threshold=0.5):
    """
    Calculate TP, FP, TN, FN given predicted and target tensors.

    Parameters:
    - predicted: Tensor containing predicted values (e.g., model output)
    - target: Tensor containing ground truth values
    - threshold: Threshold for converting probabilities to binary predictions

    Returns:
    - TP: True Positives
    - FP: False Positives
    - TN: True Negatives
    - FN: False Negatives
    """

    # Convert probabilities to binary predictions using the threshold
    binary_predictions = (predicted > threshold).float()

    # Calculate TP, FP, TN, FN
    TP = torch.sum((binary_predictions == 1) & (target == 1)).item()
    FP = torch.sum((binary_predictions == 1) & (target == 0)).item()
    TN = torch.sum((binary_predictions == 0) & (target == 0)).item()
    FN = torch.sum((binary_predictions == 0) & (target == 1)).item()

    return TP, FP, TN, FN


if __name__ == "__main__":
    main()
