from os import path as osp
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.append("..")
from Basics.RawData import RawData
from Basics.CalibData import CalibData
import Basics.params as pr
import Basics.sensorParams as psp
from Basics.Geometry import Circle
from pre_proc_images import proc_image


class RenderCalibExamples:

    def __init__(self, data_folder):
        # polytable
        calib_data = osp.join(data_folder, "polycalib.npz")
        self.calib_data = CalibData(calib_data)

        # raw calibration data
        rawData = osp.join(data_folder, "dataPack.npz")
        data_file = np.load(rawData, allow_pickle=True)
        self.f0 = proc_image(data_file['f0'])
        self.bg_proc = self.processInitialFrame()
        # cv2.imshow("bg_proc", self.bg_proc)
        # cv2.waitKey(0)
        self.imgs = data_file['imgs']
        self.radius_record = data_file['touch_radius']
        self.touchCenter_record = data_file['touch_center']

    def processInitialFrame(self):
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

        # cv2.imwrite("test_bg_proc.png", f0)

        return f0

    def render(self, idx: int):
        # keep adding items
        frame = proc_image(self.imgs[idx, :, :, :])
        # remove background
        dI = frame.astype("float") - self.bg_proc
        circle = Circle(int(self.touchCenter_record[idx, 0]), int(self.touchCenter_record[idx, 1]),
                        int(self.radius_record[idx]))

        bins = psp.numBins
        ball_radius_pix = psp.ball_radius / psp.pixmm

        center = circle.center
        radius = circle.radius

        sizey, sizex = dI.shape[:2]
        [xqq, yqq] = np.meshgrid(range(sizex), range(sizey))
        xq = xqq - center[0]
        yq = yqq - center[1]

        rsqcoord = xq * xq + yq * yq
        rad_sq = radius * radius
        # get the contact area
        valid_rad = min(rad_sq, int(ball_radius_pix * ball_radius_pix))
        valid_mask = rsqcoord < valid_rad

        # Visualize masked area.
        # img_ = frame.copy()
        # img_[~valid_mask] = 0
        # cv2.imshow("test", img_)
        # cv2.waitKey(0)

        validId = np.nonzero(valid_mask)
        xvalid = xq[validId]
        yvalid = yq[validId]
        rvalid = np.sqrt(xvalid * xvalid + yvalid * yvalid)
        # get gradients
        gradxseq_ = np.arcsin(rvalid / ball_radius_pix)
        gradxseq = np.zeros(psp.h * psp.w)
        gradxseq[valid_mask.flatten()] = gradxseq_
        gradyseq_ = np.arctan2(-yvalid, -xvalid)
        gradyseq = np.zeros(psp.h * psp.w)
        gradyseq[valid_mask.flatten()] = gradyseq_

        # generate raw simulated image without background
        sim_img_r = np.zeros((psp.h * psp.w, 3))
        bins = psp.numBins

        [xx, yy] = np.meshgrid(range(psp.w), range(psp.h))
        xf = xx.flatten()  # [valid_mask.flatten()]
        yf = yy.flatten()  # [valid_mask.flatten()]
        A = np.array([xf * xf, yf * yf, xf * yf, xf, yf, np.ones(xf.shape[0])]).T
        binm = bins - 1

        # discritize grids
        x_binr = 0.5 * np.pi / binm  # x [0,pi/2]
        y_binr = 2 * np.pi / binm  # y [-pi, pi]

        idx_x = np.floor(gradxseq / x_binr).astype('int')
        idx_y = np.floor((gradyseq + np.pi) / y_binr).astype('int')

        # look up polynomial table and assign intensity
        params_r = self.calib_data.grad_r[idx_x, idx_y, :]
        # params_r = params_r.reshape(-1, params_r.shape[2])
        params_g = self.calib_data.grad_g[idx_x, idx_y, :]
        # params_g = params_g.reshape(-1, params_g.shape[2])
        params_b = self.calib_data.grad_b[idx_x, idx_y, :]
        # params_b = params_b.reshape(-1, params_b.shape[2])

        est_r = np.sum(A * params_r, axis=1)
        est_g = np.sum(A * params_g, axis=1)
        est_b = np.sum(A * params_b, axis=1)

        sim_img_r[:, 0] = est_r
        sim_img_r[:, 1] = est_g
        sim_img_r[:, 2] = est_b
        sim_img_r = sim_img_r.reshape((psp.h, psp.w, 3))

        # attach background to simulated image
        # sim_img = sim_img_r + self.bg_proc
        sim_img = self.bg_proc + sim_img_r
        # sim_img[~valid_mask] = 255
        # sim_img = cv2.cvtColor(sim_img.astype("uint8"), cv2.COLOR_BGR2RGB)

        cv2.imshow("source", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.imshow("pred", sim_img.astype("uint8"))
        cv2.waitKey(0)


if __name__ == '__main__':
    data_folder = "/home/markvdm/RobotSetup/merl_ws/src/Taxim/data/gel/gs_mini_28N0_295H/combined_data"
    r = RenderCalibExamples(data_folder)
    for i in range(45, len(r.imgs)):
        r.render(i)
