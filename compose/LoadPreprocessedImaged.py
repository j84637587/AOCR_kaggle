import os

import torch
from monai.transforms import Compose


class LoadPreprocessedImaged(Compose):
    def __init__(self, data_dir, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir

    def __call__(self, id, **_kwargs):
        pth = torch.load(os.path.join(self.data_dir, id + ".pt"))
        return {"id": id, "image": pth["ct"], "label": pth["mask"]}


if __name__ == "__main__":
    from monai import transforms, data
    from tqdm import tqdm

    train_images = [
        "Zx0A5FF3169135AF89B21F2A97A7278E517EC8B499B0F14C20",
        "Zx0A1564C092259CD377329E07006281E876D09F9B48E77002",
        "Zx0AA2C956EEA5EAEA0DC0DA2606A8F8D0F334B97FDD860E1F",
        "Zx00AD16F8B97A53DE6E7CFE260BDF122F0E655659A3DF1628",
        "Zx0AE5424009C101F7422C5BD9DD2C3B0E13E834114A32883A",
    ]
    data_dir = "./data/preprocess/232x176x50_v5/1_Train,Valid_Image"
    data_dicts = [{"image": image_name} for image_name in zip(train_images)]
    train_transform = transforms.Compose(
        [
            LoadPreprocessedImaged(data_dir=data_dir),
            transforms.EnsureChannelFirstd(
                keys=["image", "label"], channel_dim="no_channel"
            ),
            transforms.Resized(keys=["image", "label"], spatial_size=(240, 256, 50)),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
        ]
    )
    train_ds = data.Dataset(data=train_images, transform=train_transform)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    for image in tqdm(train_loader):
        print(image["image"].shape, image["label"].shape)
    # result = train_transform(data_dicts)
    # print(result)
