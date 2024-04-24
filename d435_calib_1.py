import numpy as np
import open3d as o3d
import os
import pyrealsense2 as rs
import cv2

from enum import IntEnum
from transforms3d.quaternions import axangle2quat, qmult, quat2mat, mat2quat
from hyperparameters import *
import pickle as pkl
from pupil_apriltags import Detector
from cameras_orb import GraspCameraORB
from matplotlib import pyplot as plt
use_orb = True

class RealsesneProcessor:
    def __init__(
            self,
            first_t265_serial,
    ):
        self.first_t265_serial = first_t265_serial

        self.t265_pose = []  # pose,fisheye
        self.d435_frame = []  # rgb

        t = np.array([[-0.03083648],
                      [-0.00052044],
                      [-0.03355728]]).reshape(3)  # d435 2 t265
        R = np.array([[0.99597848, 0.08852618, -0.01378339],
                      [0.08947817, -0.99064167, 0.10306665],
                      [-0.0045303, -0.10388548, -0.99457895]])

        self.FISHEYE2POSE = np.eye(4)

        viser = o3d.visualization.Visualizer()
        viser.create_window("vis_out")
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
        self.detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=2.0,
            quad_sigma=0.8,
            refine_edges=1,
            decode_sharpening=10.0,
            debug=0
        )

        # Configure the t265 1 stream
        ctx = rs.context()
        self.t265_pipeline = rs.pipeline(ctx)
        t265_config = rs.config()
        t265_config.enable_device(self.first_t265_serial)
        t265_config.enable_stream(
            rs.stream.pose
        )
        t265_config.enable_stream(rs.stream.fisheye, 1)  # Left camera
        t265_config.enable_stream(rs.stream.fisheye, 2)  # Right camera

        self.t265_pipeline.start(t265_config)

        profiles = self.t265_pipeline.get_active_profile()
        streams = {
            "left": profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
            "right": profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile(),
        }
        self.intrinsics_t2651 = {
            "left": streams["left"].get_intrinsics(),
            "right": streams["right"].get_intrinsics(),
        }

        ex = streams["left"].get_extrinsics_to(
            profiles.get_stream(rs.stream.pose))
        R = np.asarray(ex.rotation).reshape(3, 3)
        t = np.asarray(ex.translation)
        self.FISHEYE2POSE[:3, :3] = R
        self.FISHEYE2POSE[:3, 3] = t

        self.K_t265 = np.array(
            [self.intrinsics_t2651["left"].fx, 0, self.intrinsics_t2651["left"].ppx,
             0, self.intrinsics_t2651["left"].fy, self.intrinsics_t2651["left"].ppy,
             0, 0, 1]).reshape(3, 3)
        self.D_t265 = self.intrinsics_t2651["left"].coeffs
        self.vis = None

    def get_rgbd_frame_from_realsense(self, enable_visualization=False):
        if use_orb:
            _, (color_image, depth_image) = self.pipeline.read()
            depth_image = depth_image / 1000.
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
        o3d.visualization.draw_geometries([open3d_cloud])
        return open3d_cloud

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

    def process_frame(self, i):

        t265_frames = self.t265_pipeline.wait_for_frames()

        pose_4x4 = RealsesneProcessor.frame_to_pose_conversion(
            input_t265_frames=t265_frames
        )
        left = t265_frames.get_fisheye_frame(1)
        left_data = np.asanyarray(left.get_data())

        pose = pose_4x4 @ self.FISHEYE2POSE  # pose
        print("pose:\n",pose)
        # pose2 = pose_4x4_2 @ self.c3
        flag, T,left_data = aprildetect(self.detector, left_data, self.K_t265, self.D_t265, True)
        cv2.imshow("out", left_data)
        key = cv2.waitKey(1)

        if key == ord('c'):
            self.t265_pose.append([pose_4x4, left_data])
        self.viser.clear_geometries()
        self.viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1).transform(pose_4x4))
        self.viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1).transform(pose))



        if flag:
            T1_base = pose @ T
            self.viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1).transform(T1_base))
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

    def process_frame_1(self, i, r):

        t265_frames = self.t265_pipeline.wait_for_frames()

        pose_4x4 = RealsesneProcessor.frame_to_pose_conversion(input_t265_frames=t265_frames)

        left = t265_frames.get_fisheye_frame(1)
        left_data = np.asanyarray(left.get_data())

        pose = pose_4x4 @ self.FISHEYE2POSE  # pose
        # print("FISH:",self.FISHEYE2POSE)
        flag, T,left_data = aprildetect(self.detector, left_data, self.K_t265, self.D_t265, True)

        cv2.imshow("out", left_data)
        key = cv2.waitKey(1)

        self.viser.clear_geometries()
        self.viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1).transform(pose_4x4))
        self.viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1).transform(pose))
        
        if flag:
            T_base = pose @ T
            # T_base = pose @ np.linalg.inv(T)
            self.viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1).transform(T_base))
        if r:
            pose_pre = pose
            if flag:
                pose_now = pose
                
                
       
        if i == 0:
            self.viser.run()
            ctrl = self.viser.get_view_control()
            self.params = ctrl.convert_to_pinhole_camera_parameters()

        else:
            ctrl: o3d.visualization.ViewControl = self.viser.get_view_control()
            ctrl.convert_from_pinhole_camera_parameters(self.params)

            self.viser.poll_events()

            self.params = ctrl.convert_to_pinhole_camera_parameters()
        
        return key
    
    def save(self):
        # pass
        with open("0423.pkl", 'wb') as f:
            pkl.dump({
                't265': self.t265_pose
            }, f)


import concurrent.futures


def aprildetect(detector, im, K, D, fisheye=False):
    if fisheye:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, np.asarray(D)[:4], np.eye(3), K, (848, 800), cv2.CV_16SC2)

        undistorted_img = cv2.remap(im, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    else:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
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


        for p in corners_all:
            cv2.circle(undistorted_img,(int(p[0]),int(p[1])),4,(0,255,0),-1)

        R0 = R
        t0 = tvec

        T = np.eye(4)

        T[:3, :3] = R0
        T[:3, 3] = t0.reshape(3)
        return True, T, undistorted_img
    return False, None,undistorted_img




def main():
    realsense_processor = RealsesneProcessor(
        first_t265_serial="943222111176",
    )
    realsense_processor.configure_stream()
    i = 0
    r = 0
    while True:
        key = realsense_processor.process_frame_1(i,r)
        if i == 0:
            i = 1
        if key == ord('r'):
            r = 1
        if key == ord('q'):
            break
    realsense_processor.save()


if __name__ == "__main__":
    # If user chooses to override, remove the existing directory
    main()
