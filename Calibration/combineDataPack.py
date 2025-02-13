import argparse
import os

import numpy as np

from stable_stacking.utils import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirs", type=str, nargs="+", help="Directories with data packs to combine.")
    parser.add_argument("--out_dir", "-o", type=str, help="Out directory.")
    args = parser.parse_args()
    utils.make_dir(args.out_dir)

    data_packs = [np.load(os.path.join(data_dir, "dataPack.npz"), allow_pickle=True) for data_dir in args.data_dirs]

    out_fn = os.path.join(args.out_dir, "dataPack.npz")
    np.savez(
        out_fn,
        f0=data_packs[0]['f0'],
        imgs=np.concatenate([data_pack['imgs'] for data_pack in data_packs], axis=0),
        touch_center=np.concatenate([data_pack['touch_center'] for data_pack in data_packs], axis=0),
        touch_radius=np.concatenate([data_pack['touch_radius'] for data_pack in data_packs], axis=0),
        names=np.array([f"frame_{i}.jpg" for i in
                        range(len(np.concatenate([data_pack['imgs'] for data_pack in data_packs], axis=0)))]),
        img_size=data_packs[0]['img_size'],
    )
