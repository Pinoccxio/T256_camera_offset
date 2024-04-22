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

from enum import IntEnum
from realsense_helper import get_profiles
from transforms3d.quaternions import axangle2quat, qmult, quat2mat, mat2quat
# import redis
import concurrent.futures
from hyperparameters import *
import pickle as pkl

from cameras_orb import GraspCameraORB
class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

use_orb = True

class RealsesneProcessor:
    def __init__(
            self,
            first_t265_serial,
            second_t265_serial,
            thrid_t265_serial,
    ):
        self.first_t265_serial = first_t265_serial
        self.second_t265_serial = second_t265_serial
        self.thrid_t265_serial = thrid_t265_serial

        self.t265_pose = []  # pose,fisheye
        self.d435_frame = []  # rgb

        t = np.array([[-0.03083648],
                      [-0.00052044],
                      [-0.03355728]]).reshape(3)  # d435 2 t265
        R = np.array([[0.99597848, 0.08852618, -0.01378339],
                      [0.08947817, -0.99064167, 0.10306665],
                      [-0.0045303, -0.10388548, -0.99457895]])

        viser = o3d.visualization.Visualizer()
        viser.create_window("out")
        viser.get_render_option().point_size = 4
        self.viser = viser

        self.pose = np.eye(4)
        self.pose[:3, 3] = t
        self.pose[:3, :3] = R

        between_cam_3 = np.eye(4)
        between_cam_3[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        between_cam_3[:3, 3] = np.array([0.0, -0.064, 0.0])
        self.c3 = between_cam_3

    def get_rs_t265_config(self, t265_serial, t265_pipeline):
        t265_config = rs.config()
        t265_config.enable_device(t265_serial)
        t265_config.enable_stream(rs.stream.pose)

        return t265_config

    def configure_stream(self):

        # Create a pipeline
        if use_orb:
            self.pipeline = GraspCameraORB()
        else:

            self.pipeline = rs.pipeline()
            config = rs.config()

            color_profiles, depth_profiles = get_profiles()
            w, h, fps, fmt, depin = depth_profiles[1]
            config.enable_stream(rs.stream.depth, 1280, 720, fmt, 6)
            w, h, fps, fmt, colorin = color_profiles[23]
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

            self.d435_intrinsic = colorin

        # Configure the t265 1 stream
        ctx = rs.context()
        self.t265_pipeline = rs.pipeline(ctx)
        t265_config = rs.config()
        t265_config.enable_device(self.first_t265_serial)
        t265_config.enable_stream( rs.stream.pose )

        ctx_2 = rs.context()
        self.t265_pipeline_2 = rs.pipeline(ctx_2)
        t265_config_2 = self.get_rs_t265_config(
            self.second_t265_serial, self.t265_pipeline_2
        )
        t265_config_2.enable_stream( rs.stream.pose )

        self.t265_pipeline.start(t265_config)

        self.t265_pipeline_2.start(t265_config_2)

        if not use_orb:
            pipeline_profile = self.pipeline.start(config)
            depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            depth_sensor.set_option(rs.option.visual_preset, Preset.HighDensity)
            self.depth_scale = depth_sensor.get_depth_scale()
            align_to = rs.stream.color
            self.align = rs.align(align_to)

        self.vis = None

    def get_rgbd_frame_from_realsense(self, enable_visualization=False):
        if use_orb:
            _,(color_image,depth_image) = self.pipeline.read()
            depth_image = depth_image/1000.
        else:
            frames = self.pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            depth_image = (
                    np.asanyarray(aligned_depth_frame.get_data()) / 1000.
            )  # realsense d455
            color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

    def reproject(self, rgb_image, depth_image, K):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        H, W, C = rgb_image.shape
        z = depth_image
        xi, yi = np.meshgrid(np.arange(W), np.arange(H))
        x = (xi - cx) / fx * z
        y = (yi - cy) / fy * z
        pc = np.stack([x, y, z], axis=-1)

        open3d_cloud = o3d.geometry.PointCloud()
        mask = z < 3.
        open3d_cloud.points = o3d.utility.Vector3dVector(pc[mask, :3])
        open3d_cloud.colors = o3d.utility.Vector3dVector(rgb_image[mask][:, ::-1] / 255.)
        # o3d.visualization.draw_geometries([open3d_cloud])
        return open3d_cloud

    @staticmethod
    # convert T265 frame to pose data
    def frame_to_pose_conversion(input_t265_frames):
        pose_frame = input_t265_frames.get_pose_frame()
        pose_data = pose_frame.get_pose_data()
        pose_3x3 = quat2mat(
            np.array(
                [
                    pose_data.rotation.w,
                    pose_data.rotation.x,
                    pose_data.rotation.y,
                    pose_data.rotation.z,
                ]
            )
        )
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :3] = pose_3x3
        pose_4x4[:3, 3] = [
            pose_data.translation.x,
            pose_data.translation.y,
            pose_data.translation.z,
        ]
        return pose_4x4

    def process_frame(self, i):

        t265_frames = self.t265_pipeline.wait_for_frames()
        t265_frames_2 = self.t265_pipeline_2.wait_for_frames()

        depth_frame, color_frame = self.get_rgbd_frame_from_realsense()

        if use_orb:
            K_d435 = self.pipeline.K
        else:
            K_d435 = np.array([self.d435_intrinsic.fx, 0, self.d435_intrinsic.ppx,
                           0, self.d435_intrinsic.fy, self.d435_intrinsic.ppy,
                           0, 0, 1]).reshape(3, 3)
        rgbd = self.reproject(color_frame, depth_frame, K_d435)

        # get pose data for t265 1
        pose_4x4 = RealsesneProcessor.frame_to_pose_conversion( input_t265_frames=t265_frames )


        pose_4x4_2 = RealsesneProcessor.frame_to_pose_conversion( input_t265_frames=t265_frames_2 )

        pose = pose_4x4 @ self.pose  # pose d435 2 t265

        pose2 = pose_4x4_2 @ self.c3

        cv2.imshow("out", color_frame)
        key = cv2.waitKey(1)


        self.viser.clear_geometries()
        self.viser.add_geometry(rgbd.transform(pose))
        # self.viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1))
        self.viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1).transform(pose_4x4))
        self.viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1).transform(pose2))

        if i == 0:
            #
            self.viser.run()
            # vis.update_renderer()
            ctrl = self.viser.get_view_control()
            self.params = ctrl.convert_to_pinhole_camera_parameters()

        else:
            ctrl: o3d.visualization.ViewControl = self.viser.get_view_control()
            ctrl.convert_from_pinhole_camera_parameters(self.params)

            self.viser.poll_events()

            self.params = ctrl.convert_to_pinhole_camera_parameters()

        return key

    def save(self):
        pass


# TODO:
# 去畸变（april函数中有）
# 左眼作为原点，但是需要加个offset，使得相机中心作为原点

def detect_chessboard(color_image, depth_image, objectPoints, camera_matrix, dist_coeffs):
    # 将彩色图像转换为灰度图像
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # 棋盘格内角点的行列数
    chessboard_size = (8, 6)
    
    # 查找棋盘格内角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        # 如果找到棋盘格内角点，则计算棋盘格的旋转和平移向量
        ret, rvecs, tvecs = cv2.solvePnP(objectPoints, corners, camera_matrix, dist_coeffs)
        
        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvecs)
        
        # 将旋转矩阵和平移向量合并为变换矩阵
        transform_matrix = np.hstack((rotation_matrix, tvecs))
        
        return transform_matrix

    return None

def calculate_camera_offset(reference_transform, current_transform):
    # 使用参考变换矩阵和当前变换矩阵计算相机位置偏移
    camera_offset = np.dot(np.linalg.inv(reference_transform), current_transform)
    
    return camera_offset

def visualize_camera_offset(camera_offset):
    # 可视化相机位置偏移数据，例如显示平移和旋转信息
    translation = camera_offset[:, 3]  # 提取平移向量
    rotation = camera_offset[:, :3]  # 提取旋转矩阵
    
    print("Translation vector: ", translation)
    print("Rotation matrix: ", rotation)



import concurrent.futures


def aprildetect(detector, im, K, D, fisheye=False):
    if fisheye:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (848, 800), cv2.CV_16SC2)

        undistorted_img = cv2.remap(im, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    else:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # undistorted_img = cv2.undistort(gray, K, D)
        undistorted_img = gray
    D = np.zeros_like(D)

    detects = detector.detect(undistorted_img)

    if len(detects) > 0:
        # ld,rd,rt,lt
        qs, cs = 0.0352, 0.01056
        # resolve tags info
        corners_all = []
        wps_all = []
        for det in detects:
            corners = det.corners
            id = int(det.tag_id)
            id_r = id // 6
            id_c = id % 6
            sx = id_c * (qs + cs)
            sy = id_r * (qs + cs)
            wps = np.array([[sx, sy, 0.],
                            [sx + qs, sy, 0.],
                            [sx + qs, sy + qs, 0.],
                            [sx, sy + qs, 0.]], np.float32)
            # print(wps)
            corners_all.append(corners)
            wps_all.append(wps)
        wps_all = np.concatenate(wps_all).astype(np.float32)
        corners_all = np.concatenate(corners_all).astype(np.float32)
        flag, rvec, tvec = cv2.solvePnP(wps_all, corners_all,
                                        K, D)
        R = cv2.Rodrigues(rvec)[0]

        R0 = R
        t0 = tvec

        T = np.eye(4)

        T[:3, :3] = R0
        T[:3, 3] = t0.reshape(3)
        return True, T
    return False, None


from pupil_apriltags import Detector, Detection


def calib():
    detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=2.0,
        quad_sigma=0.8,
        refine_edges=1,
        decode_sharpening=10.0,
        debug=0
    )

    realsense_processor = RealsesneProcessor(
        first_t265_serial="146322110119",
        second_t265_serial="146322110372",
        thrid_t265_serial="929122111181",
    )
    with open("1.pkl", 'rb') as f:
        data = pkl.load(f)
    realsense_processor.configure_stream()
    K_d435 = np.array([realsense_processor.d435_intrinsic.fx, 0, realsense_processor.d435_intrinsic.ppx,
                       0, realsense_processor.d435_intrinsic.fy, realsense_processor.d435_intrinsic.ppy,
                       0, 0, 1]).reshape(3, 3)
    D_d435 = realsense_processor.d435_intrinsic.coeffs

    K_t265 = np.array(
        [realsense_processor.intrinsics_t2651["left"].fx, 0, realsense_processor.intrinsics_t2651["left"].ppx,
         0, realsense_processor.intrinsics_t2651["left"].fy, realsense_processor.intrinsics_t2651["left"].ppy,
         0, 0, 1]).reshape(3, 3)
    D_t265 = realsense_processor.intrinsics_t2651["left"].coeffs

    R_d435 = []
    t_d435 = []
    R_t265 = []
    t_t265 = []

    for d435_d, t265_d in zip(data["d435"], data["t265"]):
        color = d435_d
        pose, left = t265_d
        flag, T1 = aprildetect(detector, color, K_d435, D_d435)
        T2 = pose
        if flag:
            R_d435.append(T1[:3, :3])
            t_d435.append(T1[:3, 3])
            t_t265.append(T2[:3, 3])
            R_t265.append(T2[:3, :3])

    R, t = cv2.calibrateHandEye(R_t265, t_t265, R_d435, t_d435)
    print(R, t)


def main():
    realsense_processor = RealsesneProcessor(
        first_t265_serial="146322110119",
        second_t265_serial="146322110372",
        thrid_t265_serial="929122111181",
    )
    realsense_processor.configure_stream()
    i = 0
    while True:
        key = realsense_processor.process_frame(i)
        if i == 0:
            i = 1
        if key == ord('q'):
            break
    realsense_processor.save()


if __name__ == "__main__":
    # If user chooses to override, remove the existing directory
    main()
    # calib()
