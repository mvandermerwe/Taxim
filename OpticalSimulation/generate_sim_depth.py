import argparse
import copy
import multiprocessing
import os
import time

import cv2

import arc_utilities.transformation_helper as tf_helper

import numpy as np
import trimesh
from tqdm import trange, tqdm
import tf.transformations as tr
import vedo

from stable_stacking.utils import utils, vedo_utils


# Generate "simulated" depth maps.

def get_sensor_points(imgw: int = 320, imgh: int = 240, mmpp: float = 0.065):
    mpp = mmpp / 1000.0
    x = np.arange(imgw) * mpp - (imgw / 2 * mpp)
    y = np.arange(imgh) * mpp - (imgh / 2 * mpp)
    X, Y = np.meshgrid(x, y)
    points = np.zeros([imgw * imgh, 3])
    points[:, 0] = np.ndarray.flatten(X)
    points[:, 2] = np.ndarray.flatten(Y)
    return points


def compute_depth(obj_mesh: trimesh.Trimesh, sensor_points: np.ndarray):
    ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(obj_mesh)
    min_points = copy.deepcopy(sensor_points)
    min_points[:, 1] = -0.1
    intersect_points, ray_indices, _ = ray_mesh_intersector.intersects_location(
        min_points, np.array([[0, 1, 0]] * min_points.shape[0]))
    intersect_mask = intersect_points[:, 1] < 0
    intersect_points[~intersect_mask, 1] = 0.0
    sensor_points[ray_indices] = intersect_points
    return sensor_points, intersect_mask


class GenerateSimDepth:

    def __init__(self, obj_cfg: dict, out_dir: str, num: int, length: int, seed: int):
        self.obj_cfg = obj_cfg
        self.out_dir = out_dir
        self.num = num
        self.length = length
        self.seed = seed

        self.imgw = 320
        self.imgh = 240
        self.mmpp = 0.065
        self.sensor_x_min = -(self.imgw / 2) * self.mmpp / 1000.0
        self.sensor_x_max = (self.imgw / 2) * self.mmpp / 1000.0
        self.sensor_y_min = -(self.imgh / 2) * self.mmpp / 1000.0
        self.sensor_y_max = (self.imgh / 2) * self.mmpp / 1000.0

    def __call__(self, trial_idx: int):
        seed_ = self.seed + trial_idx
        np.random.seed(seed_)

        trial_out_dir = os.path.join(self.out_dir, f"trial_{trial_idx}")
        utils.make_dir(trial_out_dir)

        # TODO: Extend pose sampling to support objects with additional rotational components.

        max_offset = np.array([0.001, 0.0005, 0.001, 4.0])  # [m, m, m, deg]
        obj_mesh = trimesh.load_mesh(self.obj_cfg["mesh"])
        sample_pose = self.obj_cfg["sample_pose"]
        obj_mesh.apply_transform(np.linalg.inv(tf_helper.BuildMatrixArray(sample_pose)))
        y_offset = np.abs(obj_mesh.bounds[0][1])

        # Get largest offset in our plane of sampling for object pose.
        obj_offset = np.max(np.abs(obj_mesh.bounds[:, [0, 2]]))

        # We will always start with a starting depth of 0.
        # This helps the model reason about no penetration.
        depth_start = np.zeros((240, 320))
        utils.save_gzip_pickle(depth_start, os.path.join(trial_out_dir, "depth_0.pkl.gzip"))
        norm_depth = cv2.normalize(depth_start, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(trial_out_dir, f"depth_0.png"), norm_depth)

        while True:
            init_obj_x = np.random.uniform(self.sensor_x_min - obj_offset,
                                           self.sensor_x_max + obj_offset)  # np.random.normal(0.0, 0.002)
            init_obj_z = np.random.uniform(self.sensor_y_min - obj_offset,
                                           self.sensor_y_max + obj_offset)  # np.random.normal(0.0, 0.002)
            init_theta = np.random.uniform(0.0, 2 * np.pi)
            indent_depth = np.random.uniform(0.0, 0.002)

            grasp_pos = np.array([init_obj_x, y_offset - indent_depth, init_obj_z])
            grasp_orn = np.array([0.0, init_theta, 0.0])
            grasp_orn_quat = tr.quaternion_from_euler(*grasp_orn)
            grasp_pose = np.concatenate([grasp_pos, grasp_orn_quat])

            # Check that the object is touching the sensor somewhere, o.w. resample.
            obj_mesh_ = copy.deepcopy(obj_mesh)
            obj_mesh_.apply_transform(tf_helper.BuildMatrixArray(grasp_pose))
            _, intersect_mask = compute_depth(obj_mesh_, get_sensor_points())
            if np.sum(intersect_mask) > 0:
                break

        current_delta = np.zeros(4)
        for step_idx in range(1, self.length + 1):
            # Sample offset!
            adjust_off_x = np.random.normal(0.0, 0.0006)
            adjust_off_y = np.random.normal(0.0, 0.0001)
            adjust_off_z = np.random.normal(0.0, 0.0006)
            adjust_theta = np.random.normal(0.0, 1.0)
            current_delta += np.stack([adjust_off_x, adjust_off_y, adjust_off_z, adjust_theta]).T
            current_delta = np.clip(current_delta, -max_offset, max_offset)

            adjust_pos = grasp_pos + current_delta[:3]
            adjust_orn = grasp_orn + np.deg2rad(np.array([0, current_delta[3], 0]))
            adjust_orn_quat = tr.quaternion_from_euler(*adjust_orn)
            adjust_pose = np.concatenate([adjust_pos, adjust_orn_quat])

            obj_mesh_trial = copy.deepcopy(obj_mesh)
            obj_mesh_trial.apply_transform(tf_helper.BuildMatrixArray(adjust_pose))

            sensor_points, _ = compute_depth(obj_mesh_trial, get_sensor_points())

            depth = np.abs(sensor_points[:, 1].reshape((240, 320)))
            utils.save_gzip_pickle(depth, os.path.join(trial_out_dir, f"depth_{step_idx}.pkl.gzip"))

            norm_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(os.path.join(trial_out_dir, f"depth_{step_idx}.png"), norm_depth)

            # vedo_plt = vedo.Plotter()
            # vedo_plt.at(0).add(
            #     vedo_utils.draw_origin(0.02), vedo.Mesh([obj_mesh_trial.vertices, obj_mesh_trial.faces], alpha=0.5),
            #     vedo.Points(sensor_points, c="blue", alpha=0.5),
            # )
            # vedo_plt.interactive().close()


def generate_sim_depth(obj_cfg: dict, out_dir: str, num: int, length: int, jobs: int = 1):
    utils.make_dir(out_dir)

    seed = int(time.time())
    gen = GenerateSimDepth(obj_cfg, out_dir, num, length, seed)

    if jobs == 1:
        for trial_idx in trange(num):
            gen(trial_idx)
    else:
        with multiprocessing.Pool(args.jobs) as pool:
            for _ in tqdm(pool.imap_unordered(gen, range(num)), total=num):
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_cfg", type=str, help="Object configuration file.")
    parser.add_argument("out_dir", type=str, help="Output directory.")
    parser.add_argument("--num", "-n", type=int, default=10, help="Number of collection trajectories to collect.")
    parser.add_argument("--length", "-l", type=int, default=5, help="Number of steps per trajectory.")
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Number of jobs.")
    args = parser.parse_args()

    obj_cfg_ = utils.load_cfg(args.obj_cfg)

    generate_sim_depth(obj_cfg_, args.out_dir, args.num, args.length, args.jobs)
