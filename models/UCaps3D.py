from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv

def _squash(input_tensor, dim=2):
    """
    Applies norm nonlinearity (squash) to a capsule layer.
    Args:
    input_tensor: Input tensor. Shape is [batch, num_channels, num_atoms] for a
        fully connected capsule layer or
        [batch, num_channels, num_atoms, height, width] or
        [batch, num_channels, num_atoms, height, width, depth] for a convolutional
        capsule layer.
    Returns:
    A tensor with same shape as input for output of this layer.
    """
    epsilon = 1e-12
    norm = torch.linalg.norm(input_tensor, dim=dim, keepdim=True)
    norm_squared = norm * norm
    return (input_tensor / (norm + epsilon)) * (norm_squared / (1 + norm_squared))

def _update_routing(votes, biases, num_routing):
    """
    Sums over scaled votes and applies squash to compute the activations.
    Iteratively updates routing logits (scales) based on the similarity between
    the activation of this layer and the votes of the layer below.
    Args:
        votes: tensor, The transformed outputs of the layer below.
        biases: tensor, Bias variable.
        num_dims: scalar, number of dimmensions in votes. For fully connected
        capsule it is 4, for convolutional 2D it is 6, for convolutional 3D it is 7.
        num_routing: scalar, Number of routing iterations.
    Returns:
        The activation tensor of the output layer after num_routing iterations.
    """
    votes_shape = votes.size()

    logits_shape = list(votes_shape)
    logits_shape[3] = 1
    logits = torch.zeros(logits_shape, requires_grad=False, device=votes.device)

    for i in range(num_routing):
        route = F.softmax(logits, dim=2)
        preactivate = torch.sum(votes * route, dim=1) + biases[None, ...]

        if i + 1 < num_routing:
            distances = F.cosine_similarity(preactivate[:, None, ...], votes, dim=3)
            logits = logits + distances[:, :, :, None, ...]
        else:
            activation = _squash(preactivate)
    return activation

class DepthwiseConv4d(nn.Module):
    """
    Performs 3D convolution given a 6D input tensor.
    This layer given an input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width, input_depth]` squeezes the
    first two dimmensions to get a 5D tensor as the input of torch.nn.Conv3d. Then
    splits the first dimmension and the second dimmension and returns the 7D
    convolution output.
    Args:
        kernel_size: scalar or tuple, convolutional kernels are [kernel_size, kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, stride of the convolutional kernel.
        padding: scalar or tuple, zero-padding added to both sides of the input
        dilation: scalar or tuple, spacing between kernel elements
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        7D Tensor output of a 3D convolution with shape
        `[batch, input_dim, output_dim, output_atoms, out_height, out_width, out_depth]`.
    """

    def __init__(
        self,
        kernel_size,
        input_dim,
        output_dim,
        input_atoms=8,
        output_atoms=8,
        stride=2,
        dilation=1,
        padding=0,
        share_weight=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_atoms = input_atoms
        self.output_atoms = output_atoms
        self.share_weight = share_weight

        if self.share_weight:
            self.conv3d = nn.Conv3d(
                input_atoms,
                output_dim * output_atoms,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            )
        else:
            self.conv3d = nn.Conv3d(
                input_dim * input_atoms,
                input_dim * output_dim * output_atoms,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                groups=input_dim,
            )
        torch.nn.init.normal_(self.conv3d.weight, std=0.1)

    def forward(self, input_tensor):
        input_shape = input_tensor.size()

        if self.share_weight:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0] * self.input_dim,
                self.input_atoms,
                input_shape[-3],
                input_shape[-2],
                input_shape[-1],
            )
        else:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0],
                self.input_dim * self.input_atoms,
                input_shape[-3],
                input_shape[-2],
                input_shape[-1],
            )

        conv = self.conv3d(input_tensor_reshaped)
        conv_shape = conv.size()

        conv_reshaped = conv.view(
            input_shape[0],
            self.input_dim,
            self.output_dim,
            self.output_atoms,
            conv_shape[-3],
            conv_shape[-2],
            conv_shape[-1],
        )
        return conv_reshaped


class DepthwiseDeconv4d(nn.Module):
    """
    Performs 3D deconvolution given a 6D input tensor.
    This layer given an input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width, input_depth]` squeezes the
    first two dimmensions to get a 5D tensor as the input of torch.nn.ConvTranspose3d. Then
    splits the first dimmension and the second dimmension and returns the 7D
    convolution output.
    Args:
        kernel_size: scalar or tuple, deconvolutional kernels are [kernel_size, kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, controls the stride for the cross-correlation.
        padding: scalar or tuple, controls the amount of implicit zero-paddings on both sides for dilation * (kernel_size - 1) - padding number of points
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        7D Tensor output of a 3D deconvolution with shape
        `[batch, input_dim, output_dim, output_atoms, out_height, out_width, out_depth]`.
    """

    def __init__(
        self,
        kernel_size,
        input_dim,
        output_dim,
        input_atoms=8,
        output_atoms=8,
        stride=2,
        padding=0,
        share_weight=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_atoms = input_atoms
        self.output_atoms = output_atoms
        self.share_weight = share_weight

        if self.share_weight:
            self.deconv3d = nn.ConvTranspose3d(
                input_atoms, output_dim * output_atoms, kernel_size, stride, padding
            )
        else:
            self.deconv3d = nn.ConvTranspose3d(
                input_dim * input_atoms,
                input_dim * output_dim * output_atoms,
                kernel_size,
                stride,
                padding,
                groups=input_dim,
            )
        torch.nn.init.normal_(self.deconv3d.weight, std=0.1)

    def forward(self, input_tensor):
        input_shape = input_tensor.size()
        if self.share_weight:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0] * self.input_dim,
                self.input_atoms,
                input_shape[-3],
                input_shape[-2],
                input_shape[-1],
            )
        else:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0],
                self.input_dim * self.input_atoms,
                input_shape[-3],
                input_shape[-2],
                input_shape[-1],
            )

        deconv = self.deconv3d(input_tensor_reshaped)
        deconv_shape = deconv.size()

        deconv_reshaped = deconv.view(
            input_shape[0],
            self.input_dim,
            self.output_dim,
            self.output_atoms,
            deconv_shape[-3],
            deconv_shape[-2],
            deconv_shape[-1],
        )
        return deconv_reshaped


class ConvSlimCapsule3D(nn.Module):
    """
    Builds a slim convolutional capsule layer.
    This layer performs 3D convolution given 6D input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width, input_depth]`. Then refines
    the votes with routing and applies Squash non linearity for each capsule.
    Each capsule in this layer is a convolutional unit and shares its kernel over
    the position grid and different capsules of layer below. Therefore, number
    of trainable variables in this layer is:
        kernel: [kernel_size, kernel_size, kernel_size, input_atoms, output_dim * output_atoms]
        bias: [output_dim, output_atoms]
    Output of a conv3d layer is a single capsule with channel number of atoms.
    Therefore conv_slim_capsule_3d is suitable to be added on top of a conv3d layer
    with num_routing=1, input_dim=1 and input_atoms=conv_channels.
    Args:
        kernel_size: scalar or tuple, convolutional kernels are [kernel_size, kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, stride of the convolutional kernel.
        padding: scalar or tuple, zero-padding added to both sides of the input
        dilation: scalar or tuple, spacing between kernel elements
        num_routing: scalar, number of routing iterations.
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        Tensor of activations for this layer of shape
        `[batch, output_dim, output_atoms, out_height, out_width, out_depth]`
    """

    def __init__(
        self,
        kernel_size,
        input_dim,
        output_dim,
        input_atoms=8,
        output_atoms=8,
        stride=2,
        padding=0,
        dilation=1,
        num_routing=3,
        share_weight=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_atoms = output_atoms
        self.num_routing = num_routing
        self.biases = nn.Parameter(
            torch.nn.init.constant_(torch.empty(output_dim, output_atoms, 1, 1, 1), 0.1)
        )
        self.depthwise_conv4d = DepthwiseConv4d(
            kernel_size=kernel_size,
            input_dim=input_dim,
            output_dim=output_dim,
            input_atoms=input_atoms,
            output_atoms=output_atoms,
            stride=stride,
            padding=padding,
            dilation=dilation,
            share_weight=share_weight,
        )

    def forward(self, input_tensor):
        votes = self.depthwise_conv4d(input_tensor)
        return _update_routing(votes, self.biases, self.num_routing)


class UCaps3D(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=4,
        share_weight=False,
        connection="skip",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.share_weight = share_weight
        self.connection = connection

        # Building model
        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        Convolution(
                            spatial_dims=3,
                            in_channels=self.in_channels,
                            out_channels=16,
                            kernel_size=5,
                            strides=1,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    (
                        "conv2",
                        Convolution(
                            spatial_dims=3,
                            in_channels=16,
                            out_channels=32,
                            kernel_size=5,
                            strides=1,
                            dilation=2,
                            padding=4,
                            bias=False,
                        ),
                    ),
                    (
                        "conv3",
                        Convolution(
                            spatial_dims=3,
                            in_channels=32,
                            out_channels=64,
                            kernel_size=5,
                            strides=1,
                            padding=4,
                            dilation=2,
                            bias=False,
                            act="tanh",
                        ),
                    ),
                ]
            )
        )

        self.primary_caps = ConvSlimCapsule3D(
            kernel_size=3,
            input_dim=1,
            output_dim=16,
            input_atoms=64,
            output_atoms=4,
            stride=1,
            padding=1,
            num_routing=1,
            share_weight=self.share_weight,
        )
        self._build_encoder()
        self._build_decoder()
        self._build_reconstruct_branch()

    def forward(self, x):
        # Contracting
        x = self.feature_extractor(x)
        x = x.unsqueeze(dim=1)
        conv_cap_1_1 = self.primary_caps(x)

        x = self.encoder_conv_caps[0](conv_cap_1_1)
        conv_cap_2_1 = self.encoder_conv_caps[1](x)

        x = self.encoder_conv_caps[2](conv_cap_2_1)
        conv_cap_3_1 = self.encoder_conv_caps[3](x)

        x = self.encoder_conv_caps[4](conv_cap_3_1)
        conv_cap_4_1 = self.encoder_conv_caps[5](x)

        shape = conv_cap_4_1.size()
        conv_cap_4_1 = conv_cap_4_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        shape = conv_cap_3_1.size()
        conv_cap_3_1 = conv_cap_3_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        shape = conv_cap_2_1.size()
        conv_cap_2_1 = conv_cap_2_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        shape = conv_cap_1_1.size()
        conv_cap_1_1 = conv_cap_1_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])

        # Expanding
        if self.connection == "skip":
            x = self.decoder_conv[0](conv_cap_4_1)
            x = torch.cat((x, conv_cap_3_1), dim=1)
            x = self.decoder_conv[1](x)
            x = self.decoder_conv[2](x)
            x = torch.cat((x, conv_cap_2_1), dim=1)
            x = self.decoder_conv[3](x)
            x = self.decoder_conv[4](x)
            x = torch.cat((x, conv_cap_1_1), dim=1)

        logits = self.decoder_conv[5](x)

        return logits

    def _build_encoder(self):
        self.encoder_conv_caps = nn.ModuleList()
        self.encoder_kernel_size = 3
        self.encoder_output_dim = [16, 16, 8, 8, 8, self.out_channels]
        self.encoder_output_atoms = [8, 8, 16, 16, 32, 64]

        for i in range(len(self.encoder_output_dim)):
            if i == 0:
                input_dim = self.primary_caps.output_dim
                input_atoms = self.primary_caps.output_atoms
            else:
                input_dim = self.encoder_output_dim[i - 1]
                input_atoms = self.encoder_output_atoms[i - 1]

            stride = 2 if i % 2 == 0 else 1

            self.encoder_conv_caps.append(
                ConvSlimCapsule3D(
                    kernel_size=self.encoder_kernel_size,
                    input_dim=input_dim,
                    output_dim=self.encoder_output_dim[i],
                    input_atoms=input_atoms,
                    output_atoms=self.encoder_output_atoms[i],
                    stride=stride,
                    padding=1,
                    dilation=1,
                    num_routing=3,
                    share_weight=self.share_weight,
                )
            )

    def _build_decoder(self):
        self.decoder_conv = nn.ModuleList()
        if self.connection == "skip":
            self.decoder_in_channels = [
                self.out_channels * self.encoder_output_atoms[-1],
                384,
                128,
                256,
                64,
                128,
            ]
            self.decoder_out_channels = [256, 128, 128, 64, 64, self.out_channels]

        for i in range(6):
            if i == 5:
                self.decoder_conv.append(
                    Conv["conv", 3](
                        self.decoder_in_channels[i],
                        self.decoder_out_channels[i],
                        kernel_size=1,
                    )
                )
            elif i % 2 == 0:
                self.decoder_conv.append(
                    UpSample(
                        spatial_dims=3,
                        in_channels=self.decoder_in_channels[i],
                        out_channels=self.decoder_out_channels[i],
                        scale_factor=2,
                    )
                )
            else:
                self.decoder_conv.append(
                    Convolution(
                        spatial_dims=3,
                        kernel_size=3,
                        in_channels=self.decoder_in_channels[i],
                        out_channels=self.decoder_out_channels[i],
                        strides=1,
                        padding=1,
                        bias=False,
                    )
                )

    def _build_reconstruct_branch(self):
        self.reconstruct_branch = nn.Sequential(
            nn.Conv3d(self.decoder_in_channels[-1], 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, self.in_channels, 1),
            nn.Sigmoid(),
        )


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UCaps3D(in_channels=1, out_channels=1).to(device)
    summary(model, input_size=(1, 1, 232, 176, 50))

    # create a sample torch feed to model
    sample_data = torch.rand(2, 1, 232, 176, 50).to(device)
    print(sample_data.shape)
    # feed the model
    model.train()
    output = model(sample_data)
    print(output.shape)
    print(output)
