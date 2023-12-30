from monai.networks.nets import UNETR
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import (
    UnetrBasicBlock,
    UnetrPrUpBlock,
    UnetrUpBlock,
)

import torch
import torch.nn as nn


class UNETR_Net(nn.Module):
    def __init__(
        self,
        output_size,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=14,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.name = "UNETR_Net"
        self.net = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed=pos_embed,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            dropout_rate=dropout_rate,
        )

        self.out = UnetOutBlock(
            spatial_dims=3,
            in_channels=14,
            out_channels=1,
        )

        self.ups = nn.Upsample(size=output_size, mode="trilinear")

    def forward(self, x):
        x = self.net(x)
        x = self.out(x)
        x = self.ups(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNETR_Net(output_size=(256, 256, 50)).to(device)

    model.net.load_state_dict(
        torch.load("./logs/unetrNet_pretrained_official/model_best.pth.tar"),
        strict=False,
    )

    summary(model, input_size=(1, 1, 96, 96, 96))

    # create a sample torch feed to model
    sample_data = torch.rand(4, 1, 96, 96, 96).to(device)
    print(sample_data.shape)
    # feed the model
    model.train()
    output = model(sample_data)
    print(output.shape)
