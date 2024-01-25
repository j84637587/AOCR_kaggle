import argparse
import os
import random
import time
import warnings
import logging
import sys

from logger import create_logger


parser = argparse.ArgumentParser(description="Training")
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
    default=1e-2,
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
    default=0.8,
    type=float,
    metavar="N",
    help="evaluate-threshold only works on evaluate (default: 0.8)",
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
    "--seed", default=1234, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--accumulation-steps",
    default=32,
    type=int,
    metavar="N",
    help="number of steps to accumulate gradients before performing optimization",
)


def main():
    args = parser.parse_args()

    args.log_path, logger = create_logger()

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

    # main_worker(args.gpu, args)


if __name__ == "__main__":
    main()
