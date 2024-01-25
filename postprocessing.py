import cv2
import numpy as np


def postprocessing(
    binary_image, image, area_thresholding=40, ct_vol_thresholding=100, logger=None
):
    total_ct_true = 0
    for slice_itr in range(binary_image.shape[-1]):
        binary_image_2d = binary_image[0, :, :, slice_itr]
        binary_image_2d = binary_image_2d.astype(np.uint8)  # Convert to CV_8U

        num_ones_before = np.count_nonzero(binary_image_2d)
        nonzero = np.nonzero(binary_image_2d)
        if len(nonzero[0]) != 0:
            # print each pixel position of nonzero
            for i in range(len(nonzero[0])):
                y = nonzero[0][i]
                x = nonzero[1][i]

                if image[0, y, x, slice_itr] <= -1000:
                    logger.info(f"AIR!!")
                    binary_image[..., slice_itr] = 0
                    break

        num_ones = np.count_nonzero(binary_image_2d)
        if num_ones_before != num_ones:
            logger.info(f"HU: \n{num_ones_before} -> {num_ones}")

        if num_ones != 0 and num_ones < area_thresholding:
            logger.info(f"num_ones: {num_ones} < {area_thresholding}")
            binary_image[0, :, :, slice_itr] = 0
        else:
            logger.info(f"num_ones: {num_ones} >= {area_thresholding}")
            total_ct_true += num_ones

    if total_ct_true < ct_vol_thresholding:
        logger.info(f"total_ct_true: {total_ct_true} < {ct_vol_thresholding}")
        binary_image = np.zeros_like(binary_image)

    logger.info(f"total_ct_true: {total_ct_true} >= {ct_vol_thresholding}")
    return binary_image


def postprocessing_old(binary_image, area_thresholding=40, connectivity=8):
    for slice_itr in range(binary_image.shape[-1]):
        binary_image_2d = binary_image[0, :, :, slice_itr]
        binary_image_2d = binary_image_2d.astype(np.uint8)  # Convert to CV_8U

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # binary_image_2d = cv2.erode(binary_image_2d, kernel, iterations=1)     # 先侵蝕，將白色小圓點移除

        # binary_image_2d = cv2.dilate(binary_image_2d, kernel)    # 再膨脹，白色小點消失

        # num_labels: 有幾個連通區域
        # labels: 每個 pixel 屬於哪個連通區域
        # stats: 每個連通區域的統計資訊, 例如: x, y, width, height, area
        # centroids: 每個連通區域的中心點
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_image_2d, connectivity, cv2.CV_32S
        )

        # print("labels: \n", labels)
        # print("stats: \n", stats)

        # 將面積小於 area_thresholding 的連通區域標記為 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < area_thresholding:
                labels[labels == i] = 0

        # 其他區域標記為 1 (闌尾炎)
        binary_image[0, :, :, slice_itr] = np.where(labels > 0, 1, 0)

        # print(binary_image[0, :, :, slice_itr])

        # print("#" * 40)
    return binary_image


def find_longest_sequence(lst):
    max_length = 0
    current_length = 0

    for num in lst:
        if num == 1:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0

    return max_length


if __name__ == "__main__":
    # create random 0, 1 with shape (1, 1, 5, 5, 3)
    binary_image = np.random.randint(2, size=(1, 256, 256, 50))

    post_processed = postprocessing(binary_image, area_thresholding=100, connectivity=4)
    print(post_processed.shape)

    import nibabel as nib

    img = nib.Nifti1Image(
        post_processed[0, ...], np.eye(4)
    )  # Save axis for data (just identity)

    img.header.get_xyzt_units()
    img.to_filename("test4d.nii")  # Save as NiBabel file
