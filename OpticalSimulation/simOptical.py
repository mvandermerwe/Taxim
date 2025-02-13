import multiprocessing
import os
from os import path as osp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import correlate
import scipy.ndimage as ndimage
from scipy import interpolate
import cv2
import argparse

import sys

from tqdm import trange, tqdm

from stable_stacking.utils import utils

sys.path.append("..")
from Basics.RawData import RawData
from Basics.CalibData import CalibData
import Basics.params as pr
import Basics.sensorParams as psp
from Calibration.pre_proc_images import proc_image, find_marker


class Renderer(object):
    def __init__(self, data_folder):
        # polytable
        calib_data = osp.join(data_folder, "polycalib.npz")
        self.calib_data = CalibData(calib_data)

        # raw calibration data
        rawData = osp.join(data_folder, "dataPack.npz")
        data_file = np.load(rawData, allow_pickle=True)
        self.f0_raw = data_file['f0']
        self.f0 = proc_image(self.f0_raw)
        self.marker_mask = find_marker(self.f0_raw, pixel_mask=1)

        self.bg_proc = self.processInitialFrame()
        # self.f0 = data_file['f0']
        # self.bg_proc = self.f0

    def processInitialFrame(self):
        """
        Smooth the initial frame
        """
        # gaussian filtering with square kernel with
        # filterSize : kscale*2+1
        # sigma      : kscale
        kscale = pr.kscale

        img_d = self.f0.astype('float')
        convEachDim = lambda in_img: gaussian_filter(in_img, kscale)

        f0 = self.f0.copy()
        for ch in range(img_d.shape[2]):
            f0[:, :, ch] = convEachDim(img_d[:, :, ch])

        frame_ = img_d

        # Checking the difference between original and filtered image
        diff_threshold = pr.diffThreshold
        dI = np.mean(f0 - frame_, axis=2)
        idx = np.nonzero(dI < diff_threshold)

        # Mixing image based on the difference between original and filtered image
        frame_mixing_per = pr.frameMixingPercentage
        h, w, ch = f0.shape
        pixcount = h * w

        for ch in range(f0.shape[2]):
            f0[:, :, ch][idx] = frame_mixing_per * f0[:, :, ch][idx] + (1 - frame_mixing_per) * frame_[:, :, ch][idx]

        return f0

    def render(self, heightMap):
        """
        Simulate the tactile image from the height map
        heightMap: heightMap of the contact

        return:
        sim_img: simulated tactile image w/o shadow
        """

        # generate gradients of the height map
        grad_mag, grad_dir = self.generate_normals(heightMap)
        depth_mask = heightMap > 0

        # generate raw simulated image without background
        sim_img_r = np.zeros((psp.h, psp.w, 3))
        bins = psp.numBins

        [xx, yy] = np.meshgrid(range(psp.w), range(psp.h))
        xf = xx.flatten()
        yf = yy.flatten()
        A = np.array([xf * xf, yf * yf, xf * yf, xf, yf, np.ones(psp.h * psp.w)]).T
        binm = bins - 1

        # discritize grids
        x_binr = 0.5 * np.pi / binm  # x [0,pi/2]
        y_binr = 2 * np.pi / binm  # y [-pi, pi]

        idx_x = np.floor(grad_mag / x_binr).astype('int')
        idx_y = np.floor((grad_dir + np.pi) / y_binr).astype('int')

        # look up polynomial table and assign intensity
        params_r = self.calib_data.grad_r[idx_x, idx_y, :]
        params_r = params_r.reshape((psp.h * psp.w), params_r.shape[2])
        params_g = self.calib_data.grad_g[idx_x, idx_y, :]
        params_g = params_g.reshape((psp.h * psp.w), params_g.shape[2])
        params_b = self.calib_data.grad_b[idx_x, idx_y, :]
        params_b = params_b.reshape((psp.h * psp.w), params_b.shape[2])

        est_r = np.sum(A * params_r, axis=1)
        est_g = np.sum(A * params_g, axis=1)
        est_b = np.sum(A * params_b, axis=1)

        sim_img_r[:, :, 0] = est_r.reshape((psp.h, psp.w))
        sim_img_r[:, :, 1] = est_g.reshape((psp.h, psp.w))
        sim_img_r[:, :, 2] = est_b.reshape((psp.h, psp.w))

        # plt.imshow(np.linalg.norm(sim_img_r, axis=-1))
        # plt.show()

        # attach background to simulated image
        sim_img = self.bg_proc + sim_img_r
        # sim_img[depth_mask] += sim_img_r[depth_mask].astype(np.uint8)
        # sim_img = self.bg_proc

        # Apply gaussian blur.
        # sim_img = cv2.GaussianBlur(sim_img.astype(np.float32), (pr.kernel_size, pr.kernel_size), 2)

        # Add markers back in.
        sim_img[self.marker_mask] = self.f0_raw[self.marker_mask]
        sim_img = cv2.GaussianBlur(sim_img.astype(np.float32), (3, 3), 2)

        return sim_img

    def interpolate(self, img):
        """
        fill the zero value holes with interpolation
        """
        x = np.arange(0, img.shape[1])
        y = np.arange(0, img.shape[0])
        # mask invalid values
        array = np.ma.masked_where(img == 0, img)
        xx, yy = np.meshgrid(x, y)
        # get the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = img[~array.mask]

        GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                   (xx, yy),
                                   method='linear', fill_value=0)  # cubic # nearest # linear
        return GD1

    def generate_normals(self, height_map):
        """
        get the gradient (magnitude & direction) map from the height map
        """
        [h, w] = height_map.shape
        top = height_map[0:h - 2, 1:w - 1]  # z(x-1,y)
        bot = height_map[2:h, 1:w - 1]  # z(x+1,y)
        left = height_map[1:h - 1, 0:w - 2]  # z(x,y-1)
        right = height_map[1:h - 1, 2:w]  # z(x,y+1)
        dzdx = (bot - top) / 2.0
        dzdy = (right - left) / 2.0

        mag_tan = np.sqrt(dzdx ** 2 + dzdy ** 2)
        grad_mag = np.arctan(mag_tan)
        invalid_mask = mag_tan == 0
        valid_mask = ~invalid_mask
        grad_dir = np.zeros((h - 2, w - 2))
        grad_dir[valid_mask] = np.arctan2(dzdx[valid_mask] / mag_tan[valid_mask],
                                          dzdy[valid_mask] / mag_tan[valid_mask])

        grad_mag = self.padding(grad_mag)
        grad_dir = self.padding(grad_dir)
        return grad_mag, grad_dir

    def padding(self, img):
        """ pad one row & one col on each side """
        return np.pad(img, ((1, 1), (1, 1)), 'symmetric')


class RenderData:

    def __init__(self, data_dir, out_dir, model_dir):
        self.renderer = Renderer(model_dir)
        self.data_dir = data_dir
        self.out_dir = out_dir

    def __call__(self, trial_idx: int):
        trial_dir = osp.join(args.data_dir, f"trial_{trial_idx}")
        trial_out_dir = osp.join(args.out_dir, f"trial_{trial_idx}")
        utils.make_dir(trial_out_dir)

        num_frames = len([f for f in os.listdir(trial_dir) if "depth_" in f and ".pkl.gzip" in f])

        nominal_depth = utils.load_gzip_pickle(os.path.join(trial_dir, "nominal_depth.pkl.gzip"))
        utils.save_gzip_pickle(nominal_depth, os.path.join(trial_out_dir, "nominal_depth.pkl.gzip"))

        for idx in range(num_frames):
            depth = utils.load_gzip_pickle(os.path.join(trial_dir, f"depth_{idx}.pkl.gzip"))
            utils.save_gzip_pickle(depth, os.path.join(trial_out_dir, f"depth_{idx}.pkl.gzip"))

            heightMap = nominal_depth[0] - depth
            utils.save_gzip_pickle(heightMap, os.path.join(trial_out_dir, f"depth_diff_{idx}.pkl.gzip"))

            heightMap *= 1e3
            heightMap /= psp.pixmm
            heightMap = cv2.GaussianBlur(heightMap.astype(np.float32), (5, 5), 10)

            sim_img = self.renderer.render(heightMap)
            cv2.imwrite(os.path.join(trial_out_dir, f"tactile_{idx}.png"), sim_img.astype(np.uint8))

            if args.vis:
                cv2.imshow("sim", sim_img.astype(np.uint8))
                cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render generate depth images.")
    parser.add_argument("data_dir", type=str, help="Data directory to render out.")
    parser.add_argument("out_dir", type=str, help="out dir")
    parser.add_argument("--model_dir", "-m", type=str,
                        default="/home/markvdm/RobotSetup/merl_ws/src/Taxim/data/gel/gs_mini_28N0_295H/combined_data",
                        help="Rendering model.")
    parser.add_argument("--vis", "-v", action="store_true", help="Visualize the output.")
    parser.set_defaults(vis=False)
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Number of jobs to run in parallel.")
    args = parser.parse_args()
    utils.make_dir(args.out_dir)

    render_data = RenderData(args.data_dir, args.out_dir, args.model_dir)

    num_trials = len([f for f in os.listdir(args.data_dir) if "trial_" in f])
    if args.jobs == 1:
        for trial_idx in trange(num_trials):
            render_data(trial_idx)
    else:
        with multiprocessing.Pool(args.jobs) as pool:
            for _ in tqdm(pool.imap_unordered(render_data, range(num_trials)), total=num_trials):
                pass
