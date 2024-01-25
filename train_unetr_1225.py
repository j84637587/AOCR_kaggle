import argparse
import os
import random
import time
import warnings
import logging

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from torchio import Compose, RandomAffine

from losses.BCEDiceLoss import BCEDiceLoss
from monai.losses import DiceCELoss
from models.UNet3d import UNet3d
from datasets.mask_dataset import build_dataloader
from trainer import Trainer
import sys

from monai.transforms import (
    AddChanneld,
    RandGaussianNoised,
    RandRotated,
    RandAffined,
    Spacingd,
)

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
    "--epochs", default=500, type=int, metavar="N", help="number of total epochs to run"
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
    default=8,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=1e-4,
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
    dest="resume",
    action="store_true",
    help="resume training from checkpoint",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--evaluate-threshold",
    default=0.5,
    type=float,
    metavar="N",
    help="evaluate-threshold only works on evaluate (default: 0.5)",
)
parser.add_argument(
    "--test-submit",
    dest="test_submit",
    action="store_true",
    help="test model on test set and generate submission file",
)
parser.add_argument(
    "--pretrained-dir",
    default="",
    type=str,
    metavar="PATH",
    help="path to pretrained model checkpoint (default: none)",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--accumulation-steps",
    default=32,
    type=int,
    metavar="N",
    help="number of steps to accumulate gradients before performing optimization",
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")


def main():
    args = parser.parse_args()

    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

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

    logger.info("=> Launch command: " + " ".join(sys.argv))
    logger.info(f"=> script: {__file__}")
    logger.info("=> args" + str(args))
    logger.info("=> creating model")
    model = UNet3d(in_channels=1, n_classes=1, n_channels=32)
    logger.info(f"=> model: {model}")

    if args.gpu is not None:
        logger.info(f"=> using gpu: {args.gpu}")
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        logger.info("=> using dataparallel")
        model = torch.nn.DataParallel(model.cuda())

    cudnn.benchmark = True

    csv_file_path = "data/TrainValid_split.csv"
    test_file_path = "data/sample_submission.csv"

    root_directory = "data/preprocess/232x176x50_v5"  # unet3d
    logger.info(f"=> root_directory {root_directory}")
    logger.info("=> loading training/validating data")

    train_transform = Compose(
        [
            AddChanneld(keys=["image", "mask"]),
        ]
    )

    valid_transform = Compose(
        [
            AddChanneld(keys=["image", "mask"]),
        ]
    )

    dataloaders = {
        "Train": build_dataloader(
            csv_file_path,
            root_directory,
            batch_size=args.batch_size,
            num_workers=args.workers,
            phase="Train",
            transform=train_transform,
        ),
        "Valid": build_dataloader(
            csv_file_path,
            root_directory,
            batch_size=args.batch_size,
            num_workers=args.workers,
            phase="Valid",
            transform=valid_transform,
        ),
        "Test": build_dataloader(
            test_file_path,
            root_directory,
            batch_size=args.batch_size,
            num_workers=args.workers,
            phase="Test",
            transform=valid_transform,
        ),
    }

    logger.info("=> start training")

    trainer = Trainer(
        net=model,
        dataloaders=dataloaders,
        criterion=DiceCELoss(),
        lr=args.lr,
        accumulation_steps=args.accumulation_steps,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        logger=logger,
        log_path=args.log_path,
        amp=False,
    )

    if args.pretrained_dir and not (args.evaluate or args.test_submit or args.resume):
        logger.error(
            "=> pretrained_dir is only used with --evaluate or --test-submit or --resume option enabled"
        )

    if args.evaluate:
        # python .\train_unetr_1225.py --batch-size 4 --pretrained-dir "./logs/2023_12_26_3" --evaluate --gen-mask
        assert (
            args.pretrained_dir
        ), "Please specify the checkpoint directory with parameter --pretrained-dir"
        trainer.load_pretrained_model(args.pretrained_dir)
        trainer.num_epochs = 1
        trainer.run(validate=True)
    elif args.test_submit:
        # python .\train_unetr_1225.py --batch-size 4 --pretrained-dir "./logs/2023_12_26_3" --evaluate-threshold 0.5 --test-submit
        assert (
            args.pretrained_dir
        ), "Please specify the checkpoint directory with parameter --pretrained-dir"
        trainer.load_pretrained_model(args.pretrained_dir)
        trainer.test_submit(threshold=args.evaluate_threshold)
    else:
        if args.resume:
            assert (
                args.pretrained_dir
            ), "Please specify the checkpoint directory with parameter --pretrained-dir"
            trainer.load_pretrained_model(args.pretrained_dir)
        trainer.run()


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
        format="%(asctime)s %(levelname)s %(message)s",
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
