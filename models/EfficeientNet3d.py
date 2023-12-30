import torch
import torch.nn as nn
from torchinfo import summary
from math import ceil


class CNNBlock3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        act=True,
        bn=True,
        bias=False,
    ):
        super(CNNBlock3d, self).__init__()
        self.cnn = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
        )  # bias set to False as we are using BatchNorm

        # if groups = in_channels then it is for Depth wise convolutional;
        # For each channel different Convolutional kernel
        # very limited change in loss but a very high decrease in number of paramteres
        # if groups = 1 : normal_conv kernel of size kernel_size**3

        self.bn = nn.BatchNorm3d(out_channels) if bn else nn.Identity()
        self.silu = nn.SiLU() if act else nn.Identity()  # SiLU <--> Swish same Thing
        # 1 layer in MBConv doesn't have activation function

    def forward(self, x):
        out = self.cnn(x)
        out = self.bn(out)
        out = self.silu(out)
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(
                1
            ),  # input C x H x W --> C x 1 X 1  ONE value of each channel
            nn.Conv3d(in_channels, reduced_dim, kernel_size=1),  # expansion
            nn.SiLU(),  # activation
            nn.Conv3d(reduced_dim, in_channels, kernel_size=1),  # brings it back
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class StochasticDepth(nn.Module):
    def __init__(self, survival_prob=0.8):
        super(StochasticDepth, self).__init__()
        self.survival_prob = survival_prob

    def forward(
        self, x
    ):  # form of dropout , randomly remove some layers not during testing
        if not self.training:
            return x
        binary_tensor = (
            torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) < self.survival_prob
        )  # maybe add 1 more here
        return torch.div(x, self.survival_prob) * binary_tensor


class MBConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio=6,
        reduction=4,  # squeeze excitation 1/4 = 0.25
        survival_prob=0.8,  # for stocastic depth
    ):
        super(MBConv3d, self).__init__()

        self.survival_prob = 0.8
        self.use_residual = (
            in_channels == out_channels and stride == 1
        )  # Important if we downsample then we can't use skip connections
        hidden_dim = int(in_channels * expand_ratio)
        self.expand = in_channels != hidden_dim  # every first layer in MBConv
        reduced_dim = int(in_channels / reduction)
        self.padding = padding

        # expansion phase

        self.expand = (
            nn.Identity()
            if (expand_ratio == 1)
            else CNNBlock3d(in_channels, hidden_dim, kernel_size=1)
        )

        # Depthwise convolution phase
        self.depthwise_conv = CNNBlock3d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=hidden_dim,
        )

        # Squeeze Excitation phase
        self.se = SqueezeExcitation(hidden_dim, reduced_dim=reduced_dim)

        # output phase
        self.pointwise_conv = CNNBlock3d(
            hidden_dim, out_channels, kernel_size=1, stride=1, act=False, padding=0
        )
        # add Sigmoid Activation as mentioned in the paper

        # drop connect
        self.drop_layers = StochasticDepth(survival_prob=survival_prob)

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.pointwise_conv(x)

        if self.use_residual:  # and self.depthwise_conv.stride[0] == 1:
            x = self.drop_layers(x)
            x += residual
        return x


class EfficeientNet3d(nn.Module):
    def __init__(self, width_mult=1, depth_mult=1, dropout_rate=0.1, num_classes=2):
        super(EfficeientNet3d, self).__init__()
        last_channels = ceil(512 * width_mult)

        self.first_layer = CNNBlock3d(1, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool3d(1, stride=2)
        self.features = self._feature_extractor(width_mult, depth_mult, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels * 8 * 6 * 2, 400),
            nn.Linear(400, 64),
            nn.Linear(
                64, num_classes
            ),  # Adjust the output size based on the number of classes
        )
        self.sigmoid = nn.Sigmoid()

    def _feature_extractor(self, width_mult, depth_mult, last_channel):
        # Your previous code for scaling channels and layers

        layers = []
        in_channels = 64  # Initial input channels after the first layer
        final_in_channel = 0  # Initialzse

        # Define configurations for the custom MBConv blocks
        mbconv_configurations = [
            (3, 1, 64, 64, 1),
            (5, 2, 64, 96, 1),
            (5, 2, 96, 128, 2),
            (5, 2, 128, 192, 3),
            (3, 1, 192, 256, 1),
        ]

        for (
            kernel_size,
            stride,
            in_channels,
            out_channels,
            repeats,
        ) in mbconv_configurations:
            layers += [
                MBConv3d(
                    in_channels if repeat == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride if repeat == 0 else 1,
                    expand_ratio=1,  # Assuming you want expansion factor 1 for these blocks
                    padding=kernel_size // 2,
                )
                for repeat in range(repeats)
            ]
            final_in_channel = out_channels
            print(
                f"in_channels : {in_channels}, out_channels: {out_channels}, "
                f"kernelsize : {kernel_size}, stride: {stride}, repeats: {repeats}"
            )
        #         print(f'final_in_channels : {final_in_channel}')
        layers.append(
            MBConv3d(final_in_channel, last_channel, kernel_size=1, stride=1, padding=0)
        )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        out = self.first_layer(inputs)  # 3X3 -> BN -> SiLU
        out = self.pool(out)  # maxpool
        x = self.features(out)
        # print(x.shape)
        dummy = x.view(x.shape[0], -1)  # flatten

        out = self.classifier(dummy)
        out = self.sigmoid(out)
        return out


if __name__ == "__main__":
    import numpy as np

    dropout_rate = 0.1
    model = EfficeientNet3d(num_classes=50, dropout_rate=dropout_rate)
    batch_size = 12

    print(summary(model, input_size=(batch_size, 1, 232, 176, 50)))

    sample_data = np.random.rand(batch_size, 1, 232, 176, 50).astype(np.float32)
    sample_tensor = torch.from_numpy(sample_data)
    target = np.random.randint(2, size=(batch_size, 50))
    target = torch.tensor(target, dtype=torch.long)

    model.train()
    output = model(sample_tensor)
    print(output.shape)
