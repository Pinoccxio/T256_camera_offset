"""
Example usage

python data_recording.py -s --store_hand -o ./save_data_scenario_1
"""

import argparse
import copy
import numpy as np
import open3d as o3d
import os
import shutil
import sys
import pyrealsense2 as rs
import cv2
import time

from enum import IntEnum
from realsense_helper import get_profiles
from transforms3d.quaternions import axangle2quat, qmult, quat2mat, mat2quat

import concurrent.futures
from hyperparameters import *
import h5py
from cameras import RealsenseProcessor
import signal
from traceback import print_exc

from datetime import datetime

# def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
#     max_idx = 1000
#     if not os.path.isdir(dataset_dir):
#         os.makedirs(dataset_dir)
#     for i in range(max_idx+1):
#         if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
#             return i
#     raise Exception(f"Error getting auto index, or more than {max_idx} episodes")
from scipy.spatial.transform import Rotation


def toT(msg):
    t = msg[:3]
    q = msg[3:]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


class calibeyeglove:
    def __init__(self):
        self.R_glove = []
        self.t_glove = []
        self.R_t265 = []
        self.t_t265 = []

    def insert(self, T_glove, T_t265):
        T_t265 = np.linalg.inv(T_t265)  # t265base 2 t265now
        self.R_glove.append(T_glove[:3, :3])
        self.t_glove.append(T_glove[:3, 3])
        self.R_t265.append(T_t265[:3, 3])
        self.t_t265.append(T_t265[:3, :3])

    def __len__(self):
        return len(self.R_glove)

    def calib(self):
        # R glove 2 t265   bTt tTg = bTg

        R, t = cv2.calibrateHandEye(self.R_glove, self.t_glove, self.R_t265, self.t_t265)

        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t.reshape(3)
        print("calib", T)
        return T


def main():
    realsense_processor = RealsenseProcessor(
        first_t265_serial="146322110119",
        second_t265_serial="146322110372",
        third_t265_serial="929122111181",
        save_hand=True
    )
    exit_program = False

    viser = o3d.visualization.Visualizer()
    viser.create_window("out")
    viser.get_render_option().point_size = 4

    def signal_handler(signal, frame):
        exit_program = True

    signal.signal(signal.SIGINT, signal_handler)

    # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    # encode_rgb_length_max = 0
    # encode_rgb_length = []
    # encode_depth_length_max = 0
    # encode_depth_length = []
    max_length = 10000

    realsense_processor.configure_stream()
    tips = [
        "thumb",
        "index",
        "middle",
        "ring",
        "little"]
    length = 0

    # dataset_path = args.log_dir
    # os.makedirs(dataset_path, exist_ok=True)
    # filename = f"episode_{get_auto_index(dataset_path)}.hdf5"

    FPS = 20
    time0 = time.time()
    DT = 1 / FPS
    settime = None
    i = 0
    # try:

    tTg = None

    calib = calibeyeglove()

    while not exit_program and length < max_length:
        t0 = time.time()
        data = realsense_processor.process_frame()
        # rgb, depth = data["d435"]

        hand_params = np.concatenate(
            [data["hand"]["left"][i + 1] for i in range(20)] + [data["hand"]["lefttip"][tip] for tip in tips] + \
            [data["hand"]["right"][i + 1] for i in range(20)] + [data["hand"]["righttip"][tip] for tip in tips])

        viser.clear_geometries()
        # viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1))

        pose2 = toT(data["t2652"])
        hand_pose = toT(data["hand"]["right"][1])

        if tTg is None:
            if i > 1:
                calib.insert(hand_pose, pose2)

                if len(calib) == 10:
                    tTg = calib.calib()

            viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.5).transform(hand_pose))
            viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.5).transform(pose2))

        else:
            viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.5).transform(pose2))
            viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.5).transform(pose2 @ tTg))

        if i == 0:
            #
            viser.run()
            # vis.update_renderer()
            ctrl = viser.get_view_control()
            params = ctrl.convert_to_pinhole_camera_parameters()

        else:
            ctrl: o3d.visualization.ViewControl = viser.get_view_control()
            ctrl.convert_from_pinhole_camera_parameters(params)

            viser.poll_events()

            params = ctrl.convert_to_pinhole_camera_parameters()
        i += 1

        # cv2.imshow("out",rgb)
        # cv2.imshow("depth", cv2.applyColorMap((depth / 1000. * 255).astype(np.uint8), cv2.COLORMAP_JET))
        # key = cv2.waitKey(1)

    # except Exception as e:
    #    print_exc(e)
    # finally:


if __name__ == "__main__":
    main()
