import argparse
import os
import random
import time
import warnings
import json
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms

import matplotlib.pyplot as plt

from losses.BCEDiceLoss import BCEDiceLoss
from models.UNet3d import UNet3d
from datasets.mask_dataset import build_dataloader
from trainer import Trainer

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
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")


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

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu
    args.log_path, logger = create_logger()

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    # create model
    logger.info("=> args" + str(args))
    logger.info("=> creating model")
    model = UNet3d(in_channels=1, n_classes=1, n_channels=16)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         logger.info("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint["epoch"]
    #         best_acc1 = checkpoint["best_acc1"]
    #         if args.gpu is not None:
    #             # best_acc1 may be from a checkpoint from a different GPU
    #             best_acc1 = best_acc1.to(args.gpu)
    #         model.load_state_dict(checkpoint["state_dict"])
    #         optimizer.load_state_dict(checkpoint["optimizer"])
    #         logger.info(
    #             "=> loaded checkpoint '{}' (epoch {})".format(
    #                 args.resume, checkpoint["epoch"]
    #             )
    #         )
    #     else:
    #         logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    csv_file_path = "data/TrainValid_split.csv"
    root_directory = "data/preprocess/232x176x50_m"
    logger.info(f"=> root_directory {root_directory}")

    logger.info("=> loading training/validating data")
    # transform = transforms.Compose([permute_and_expand_with_zeros])
    dataloaders = {
        "Train": build_dataloader(
            csv_file_path,
            root_directory,
            batch_size=args.batch_size,
            num_workers=args.workers,
            phase="Train",
        ),
        "Valid": build_dataloader(
            csv_file_path,
            root_directory,
            batch_size=args.batch_size,
            num_workers=args.workers,
            phase="Valid",
        ),
    }

    logger.info("=> start training")

    trainer = Trainer(
        net=model,
        dataloaders=dataloaders,
        criterion=BCEDiceLoss(),
        lr=5e-4,
        accumulation_steps=4,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        logger=logger,
        log_path=args.log_path,
        display_plot=True,
    )

    trainer.run()


def plot_history(history, log_path, show=False):
    f, ax = plt.subplots(2, 3, figsize=(15, 5))
    f.subplots_adjust(hspace=0.5)  # Add gap between ax

    for i, name in enumerate(["loss", "acc", "dice_loss"]):
        ax[0][i].set_title(name)
        ax[0][i].plot(history["Train"][name], label=f"train {name}")
        ax[0][i].plot(history["Valid"][name], label=f"val {name}")
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


if __name__ == "__main__":
    main()
