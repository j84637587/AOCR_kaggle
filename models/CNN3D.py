import torch
import torch.nn as nn
import numpy as np


class CNN3D(nn.Module):
    def __init__(self, width=128, height=128, depth=64):
        super(CNN3D, self).__init__()

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.batch_norm1 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.batch_norm2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        self.batch_norm3 = nn.BatchNorm3d(128)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=0)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        self.batch_norm4 = nn.BatchNorm3d(256)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, depth)  # output size is the same as depth
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.batch_norm1(self.pool1(nn.functional.relu(self.conv1(x))))
        x = self.batch_norm2(self.pool2(nn.functional.relu(self.conv2(x))))
        x = self.batch_norm3(self.pool3(nn.functional.relu(self.conv3(x))))
        x = self.batch_norm4(self.pool4(nn.functional.relu(self.conv4(x))))

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = CNN3D(width=512, height=512, depth=50)

    print(summary(model, input_size=(10, 1, 232, 176, 50)))

    sample_data = np.random.rand(4, 1, 256, 256, 50).astype(np.float32)
    sample_tensor = torch.from_numpy(sample_data)
    target = np.random.randint(2, size=(4, 50))
    target = torch.tensor(target, dtype=torch.long)

    print(target.shape)

    model.train()
    output = model(sample_tensor)

    def accuracy_binary(output, target):
        """Computes the binary accuracy"""
        with torch.no_grad():
            pred = torch.round(torch.sigmoid(output))
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
            print(pred)

            correct = target.eq(pred.view_as(target)).sum().item()
            total = target.size(0) * target.size(1)
            acc = correct / total
            return acc

    print(accuracy_binary(output, target))
