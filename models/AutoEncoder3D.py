import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, in_channel=4):
        super(AutoEncoder, self).__init__()
        self.in_channel = in_channel
        # Encoder
        self.conv1 = nn.Conv3d(self.in_channel, 16, 3)
        self.conv2 = nn.Conv3d(16, 32, 3)
        self.conv3 = nn.Conv3d(32, 96, 2)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3, return_indices=True)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.enc_linear = nn.Linear(67392, 512)

        # Decoder
        self.deconv1 = nn.ConvTranspose3d(96, 32, 2)
        self.deconv2 = nn.ConvTranspose3d(32, 16, 3)
        self.deconv3 = nn.ConvTranspose3d(16, self.in_channel, 3)
        self.unpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unpool2 = nn.MaxUnpool3d(kernel_size=3, stride=3)
        self.unpool3 = nn.MaxUnpool3d(kernel_size=2, stride=2)

        self.dec_linear = nn.Linear(512, 67392)

    def encode(self, x, return_partials=True):
        # Encoder
        x = self.conv1(x)
        up3out_shape = x.shape
        # print(up3out_shape)
        x, indices1 = self.pool1(x)
        x = self.conv2(x)
        up2out_shape = x.shape
        # print(up2out_shape)
        x, indices2 = self.pool2(x)
        x = self.conv3(x)
        up1out_shape = x.shape
        # print(up1out_shape)
        x, indices3 = self.pool3(x)
        # print(x.shape)
        x = x.view((x.size(0), -1))
        # print(x.shape)
        x = self.enc_linear(x)
        # print(x.shape)

        # required for unpool
        pool_par = {
            "P1": [indices1, up3out_shape],
            "P2": [indices2, up2out_shape],
            "P3": [indices3, up1out_shape],
        }

        if return_partials:
            return x, pool_par
        else:
            return x

    def decode(self, x, pool_par):
        x = self.dec_linear(x)
        # print(x.shape)
        # x = x.view((x.size(0), 96, 11, 19, 19))
        x = x.view((x.size(0), 96, 3, 18, 13))
        # print(x.shape)

        x = self.unpool1(x, output_size=pool_par["P3"][1], indices=pool_par["P3"][0])
        # print(x.shape)
        x = self.deconv1(x)
        # print(x.shape)
        x = self.unpool2(x, output_size=pool_par["P2"][1], indices=pool_par["P2"][0])
        # print(x.shape)
        x = self.deconv2(x)
        # print(x.shape)
        x = self.unpool3(x, output_size=pool_par["P1"][1], indices=pool_par["P1"][0])
        # print(x.shape)
        x = self.deconv3(x)
        # print(x.shape)
        return x

    def forward(self, x):
        self.feature, pool_par = self.encode(x)
        out = self.decode(self.feature, pool_par)
        return out


if __name__ == "__main__":
    input = torch.rand(2, 1, 50, 232, 176)
    # input = torch.rand(1, 4, 155, 240, 240)
    model = AutoEncoder(in_channel=1)
    out = model(input)
    print("Done.\n Final out shape is: ", out.shape)

"""
torch.Size([1, 16, 153, 238, 238])
torch.Size([1, 32, 74, 117, 117])
torch.Size([1, 96, 23, 38, 38])
torch.Size([1, 96, 11, 19, 19])
torch.Size([1, 381216])
torch.Size([1, 512])
torch.Size([1, 381216])
torch.Size([1, 96, 11, 19, 19])
torch.Size([1, 96, 23, 38, 38])
"""
