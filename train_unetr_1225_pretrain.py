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
from torchvision import transforms
from torchio import Compose, RandomAffine

from losses.BCEDiceLoss import BCEDiceLoss
from models.UNETR_official import UNETR_Net
from datasets.mask_dataset import build_dataloader
from trainer import Trainer
import sys

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
    default=4,
    type=int,
    metavar="N",
    help="number of steps to accumulate gradients before performing optimization",
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

    logger.info("=> Launch command: " + " ".join(sys.argv))
    logger.info(f"=> script: {__file__}")
    logger.info("=> args" + str(args))
    logger.info("=> creating model")
    model = UNETR_Net(output_size=(240, 176, 48))
    # model = UNet3d(in_channels=1, n_classes=1, n_channels=32)
    logger.info(f"=> model: {model}")

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    csv_file_path = "data/TrainValid_split.csv"
    test_file_path = "data/sample_submission.csv"

    root_directory = "data/preprocess/240x176x48"  # unetr
    # root_directory = "data/preprocess/232x176x50_v3"  # unet3d
    logger.info(f"=> root_directory {root_directory}")

    logger.info("=> loading training/validating data")

    transform = Compose(
        [
            RandomAffine(
                degrees=(2),
                translation=(2 / 512, 2 / 512),
                scales=(0.001),
            )
        ]
    )
    logger.info(f"=> transform: {transform}")

    dataloaders = {
        "Train": build_dataloader(
            csv_file_path,
            root_directory,
            batch_size=args.batch_size,
            num_workers=args.workers,
            phase="Train",
            transform=transform,
        ),
        "Valid": build_dataloader(
            csv_file_path,
            root_directory,
            batch_size=args.batch_size,
            num_workers=args.workers,
            phase="Valid",
        ),
        "Test": build_dataloader(
            test_file_path,
            root_directory,
            batch_size=args.batch_size,
            num_workers=args.workers,
            phase="Test",
        ),
    }

    logger.info("=> start training")

    trainer = Trainer(
        net=model,
        dataloaders=dataloaders,
        criterion=BCEDiceLoss(),
        lr=5e-4,
        accumulation_steps=args.accumulation_steps,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        logger=logger,
        log_path=args.log_path,
        display_plot=True,
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
        # python .\train_unetr_1225.py --batch-size 4 --pretrained-dir "./logs/2023_12_26_3" --test_submit
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
            logger.info(f"=> load pretrained from {args.pretrained_dir}")
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
