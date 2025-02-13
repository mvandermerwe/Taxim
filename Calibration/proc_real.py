import argparse
import os
import cv2

from pre_proc_images import proc_image
from stable_stacking.utils import utils


def pre_proc_images(data_folder: str, out_dir: str = None):
    utils.make_dir(args.out_dir)
    num_images = len([f for f in os.listdir(data_folder) if "tactile_" in f and ".png" in f])

    for image_idx in range(num_images):
        image_path = os.path.join(data_folder, f"tactile_{image_idx}.png")
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = proc_image(img)

        cv2.imshow("raw", img)
        cv2.waitKey(0)

        if out_dir is not None:
            cv2.imwrite(os.path.join(out_dir, f"tactile_{image_idx}.png"), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Data directory.")
    parser.add_argument("--out_dir", "-o", type=str, default=None, help="Out directory.")
    args = parser.parse_args()

    pre_proc_images(args.data_dir, args.out_dir)