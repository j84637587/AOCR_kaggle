import torch


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    return checkpoint


# Replace 'filepath' with the actual path to your checkpoint file
checkpoint_filepath = r"F:\AOCR\logs\2023_12_18_7\model_best.pth.tar"
checkpoint = load_checkpoint(checkpoint_filepath)

# Print the information of the checkpoint
print(checkpoint["epoch"])
print(checkpoint["best_acc1"])
