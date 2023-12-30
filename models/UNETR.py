from monai.networks.nets import UNETR

import torch
import torch.nn as nn


class UNETR_Net(nn.Module):
    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=1,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
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
            res_block=res_block,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNETR_Net(img_size=(240, 176, 48)).to(device)
    summary(model, input_size=(1, 1, 240, 176, 48))

    # create a sample torch feed to model
    sample_data = torch.rand(4, 1, 240, 176, 48).to(device)
    print(sample_data.shape)
    # feed the model
    model.train()
    output = model(sample_data)
    print(output.shape)
    print(output)
