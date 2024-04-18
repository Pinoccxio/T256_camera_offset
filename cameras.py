import copy
import numpy as np
import os
import shutil
import sys
import pyrealsense2 as rs
import cv2

from enum import IntEnum
# from realsense_helper import get_profiles
from transforms3d.quaternions import axangle2quat, qmult, quat2mat, mat2quat
from hyperparameters import *
from socket_test import SocketInterface
from scipy.spatial.transform import Rotation


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def get_profiles(sw=1280, sh=720, depth_format=rs.format.z16, rgb_format=rs.format.bgr8):
    ctx = rs.context()
    devices = ctx.query_devices()

    # color_profiles = []
    # depth_profiles = []
    fpsmax_rgb = 0.
    fpsmax_depth = 0.
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)

        for sensor in device.query_sensors():
            for stream_profile in sensor.get_stream_profiles():
                stream_type = str(stream_profile.stream_type())

                if stream_type in ['stream.color', 'stream.depth']:
                    v_profile = stream_profile.as_video_stream_profile()
                    fmt = stream_profile.format()
                    w, h = v_profile.width(), v_profile.height()
                    fps = v_profile.fps()

                    # intrinsic = v_profile.get_intrinsics()

                    video_type = stream_type.split('.')[-1]
                    # print('  {}: width={}, height={}, fps={}, fmt={}'.format(
                    #     video_type, w, h, fps, fmt),intrinsic)

                    if video_type == 'color' and (w == sw and h == sh and fmt == rgb_format):
                        if fps > fpsmax_rgb:
                            fpsmax_rgb = fps
                    elif (w == sw and h == sh and fmt == depth_format):
                        if fps > fpsmax_depth:
                            fpsmax_depth = fps

                    # if video_type == 'color' and (w == sw and h == sh and fmt == rgb_format):
                    #     color_profiles.append((w, h, fps, fmt, intrinsic))
                    # elif (w == sw and h == sh and fmt == rgb_format):
                    #     depth_profiles.append((w, h, fps, fmt, intrinsic))

    return fpsmax_rgb, fpsmax_depth


class RealsenseProcessor:
    def __init__(
            self,
            first_t265_serial,
            second_t265_serial,
            third_t265_serial,
            save_hand=True,
    ):
        self.first_t265_serial = first_t265_serial
        self.second_t265_serial = second_t265_serial
        self.third_t265_serial = third_t265_serial
        self.save_hand = save_hand
        if self.save_hand:
            self.hand_server = SocketInterface()

    def get_rs_t265_config(self, t265_serial, t265_pipeline):
        t265_config = rs.config()
        t265_config.enable_device(t265_serial)
        t265_config.enable_stream(rs.stream.pose)

        return t265_config

    def configure_stream(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()

        # color_profiles, depth_profiles = get_profiles()
        fps_rgb, fps_depth = get_profiles()
        fps = min(fps_rgb, fps_depth)
        # w, h, fps, fmt, depin = depth_profiles[1]
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)
        # w, h, fps, fmt, colorin = color_profiles[23]
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, fps)

        ctx = rs.context()
        self.t265_pipeline = rs.pipeline(ctx)
        t265_config = rs.config()
        t265_config.enable_device(self.first_t265_serial)
        t265_config.enable_stream(
            rs.stream.pose
        )

        ctx_2 = rs.context()
        self.t265_pipeline_2 = rs.pipeline(ctx_2)
        t265_config_2 = self.get_rs_t265_config(
            self.second_t265_serial, self.t265_pipeline_2
        )
        t265_config_2.enable_stream(rs.stream.pose)

        # try:
        #     # Configure the t265 3 stream
        #     ctx_3 = rs.context()
        #     self.t265_pipeline_3 = rs.pipeline(ctx_3)
        #     t265_config_3 = self.get_rs_t265_config(
        #         self.thrid_t265_serial, self.t265_pipeline_3
        #     )
        # except:
        #     pass

        self.t265_pipeline.start(t265_config)

        # try:
        self.t265_pipeline_2.start(t265_config_2)
        # self.t265_pipeline_3.start(t265_config_3)

        pipeline_profile = self.pipeline.start(config)
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, Preset.Custom)
        self.depth_scale = depth_sensor.get_depth_scale()
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_rgbd_frame_from_realsense(self, enable_visualization=False):
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

    @staticmethod
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

    def process_frame(self):
        t265_frames = self.t265_pipeline.wait_for_frames()
        t265_frames_2 = self.t265_pipeline_2.wait_for_frames()

        depth_frame, color_frame = self.get_rgbd_frame_from_realsense()
        if self.save_hand:
            hand_cfg = self.hand_server.wait_for_frame()

        # get pose data for t265 1
        pose_4x4 = RealsenseProcessor.frame_to_pose_conversion(
            input_t265_frames=t265_frames
        )
        # left = t265_frames.get_fisheye_frame(1)
        # left_data = np.asanyarray(left.get_data())

        pose_4x4_2 = RealsenseProcessor.frame_to_pose_conversion(
            input_t265_frames=t265_frames_2
        )

        # pose = pose_4x4 @ self.pose  # pose d435 2 t265

        # pose2 = self.c3 @ pose_4x4_2

        data_return = {
            't2651': np.concatenate([pose_4x4[:3, 3], Rotation.from_matrix(pose_4x4[:3, :3]).as_quat()]),
            't2652': np.concatenate([pose_4x4_2[:3, 3], Rotation.from_matrix(pose_4x4_2[:3, :3]).as_quat()]),
            'd435': [color_frame, depth_frame],
        }
        if self.save_hand:
            data_return.update({'hand': hand_cfg})

        return data_return

    def close(self):
        self.t265_pipeline.stop()
        self.t265_pipeline_2.stop()
        self.pipeline.stop()
        if self.save_hand:
            self.hand_server.close()
