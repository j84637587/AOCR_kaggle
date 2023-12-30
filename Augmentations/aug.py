import albumentations as A
import cv2


def get_train_transforms(image_size):
    return A.Compose(
        [
            A.RandomContrast(limit=0.2, p=1.0),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0,
            ),
            A.Cutout(
                num_holes=2,
                max_h_size=int(0.4 * image_size),
                max_w_size=int(0.4 * image_size),
                fill_value=0,
                always_apply=True,
                p=1.0,
            ),
        ]
    )
