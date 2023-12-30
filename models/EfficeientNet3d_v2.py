import torch.nn as nn
from efficientnet_pytorch_3d import EfficientNet3D


class EfficeientNet3d(nn.Module):
    def __init__(self, num_classes=50, in_channels=1, name="efficientnet-b0"):
        super().__init__()
        self.net = EfficientNet3D.from_name(
            name,
            override_params={"num_classes": num_classes},
            in_channels=in_channels,
        )

        self.net._fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(
                in_features=self.net._fc.in_features,
                out_features=num_classes,
                bias=True,
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.net(x)
        out = self.sigmoid(out)  # make sure the output is between 0 and 1
        return out


if __name__ == "__main__":
    # import numpy as np
    from torchinfo import summary
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficeientNet3d(num_classes=50, in_channels=1).to(device)
    batch_size = 12

    summary(model, input_size=(12, 1, 256, 64, 256))

    sample_data = torch.rand((batch_size, 1, 256, 64, 256)).float().to(device)
    target = torch.randint(2, size=(batch_size, 50, 1)).to(device)

    print(sample_data.shape)
    print(target.shape)
    out = model(sample_data)
    print(out.shape)
    print(out)

    criterion = nn.BCEWithLogitsLoss().to(device)
    loss = criterion(out, target)
    print(loss)
