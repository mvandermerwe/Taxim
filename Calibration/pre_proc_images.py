import os

import cv2
import numpy as np
from scipy.ndimage import binary_dilation


def grow_mask(mask, pixels):
    """
    Expands each masked region in the given mask by a specified number of pixels.

    Parameters:
        mask (ndarray): 2D NumPy array representing the mask (1 for masked, 0 for background).
        pixels (int): Number of pixels to expand the mask.

    Returns:
        ndarray: Expanded mask.
    """
    structuring_element = np.ones((3, 3))  # 8-connectivity kernel
    grown_mask = mask.copy()

    for _ in range(pixels):
        grown_mask = binary_dilation(grown_mask, structure=structuring_element)

    return grown_mask


def find_marker(img, pixel_mask: int = 3):
    # img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # mask = cv2.inRange(img_, 0, 70) > 0

    # Check for colors close to black.
    black = np.array([0, 0, 0])

    mask = np.linalg.norm(img - black, axis=-1) < 110
    mask = grow_mask(mask, pixel_mask)
    return mask


def proc_image(img):
    marker_mask = find_marker(img)

    # img[marker_mask] = 255
    # cv2.imshow("marker", img)

    # Inpaint the markers.
    img_ = cv2.inpaint(img, marker_mask.astype(np.uint8), 20, cv2.INPAINT_NS)

    # cv2.imshow("inpaint", img_)
    # cv2.waitKey(0)

    return img_


def pre_proc_images(data_folder: str):
    num_images = len([f for f in os.listdir(data_folder) if "frame_" in f and ".jpg" in f])

    for image_idx in range(num_images):
        image_path = os.path.join(data_folder, f"frame_{image_idx}.jpg")
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = proc_image(img)

        cv2.imshow("raw", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    data_folder = "/home/markvdm/RobotSetup/merl_ws/src/Taxim/data/gel/gs_mini_28N0_295H/02_11_2025/sphere_4mm"
    pre_proc_images(data_folder)
