"""
Visualize saved pcd file and poses

Example usage:
(1) for all frames
python replay_human_traj_vis.py --directory ./saved_data/

(2) for calibration
python replay_human_traj_vis.py --directory ./saved_data/ -calib
"""

import argparse
import os
import copy
import zmq
import cv2
import sys
import shutil
import open3d as o3d
import numpy as np
import platform
import h5py
from scipy.spatial.transform import Rotation

lines = np.array([
    # Thumb
    [1, 2], [2, 3],
    # Index
    [4, 5], [5, 6], [6, 7],
    # Middle
    [8, 9], [9, 10], [10, 11],
    # Ring
    [12, 13], [13, 14], [14, 15],
    # Little
    [16, 17], [17, 18], [18, 19],
    # Connections between proximals
    [1, 4], [4, 8], [8, 12], [12, 16],
    # connect palm
    [0, 1], [16, 0]
])
Distal_map = {
    3: "thumb",
    7: "index",
    11: "middle",
    15: "ring",
    19: "little"
}
tips = [
    "thumb",
    "index",
    "middle",
    "ring",
    "little"]


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
        self.R_t265.append(T_t265[:3, :3])
        self.t_t265.append(T_t265[:3, 3])

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


def create_or_update_cylinder(start, end, radius=0.003, cylinder_list=None, cylinder_idx=-1):
    # Calculate the length of the cylinder
    cyl_length = np.linalg.norm(end - start)

    # Create a new cylinder
    new_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=cyl_length, resolution=20, split=4)
    new_cylinder.paint_uniform_color([1, 0, 0])

    new_cylinder.translate(np.array([0, 0, cyl_length / 2]))

    # Compute the direction vector of the line segment and normalize it
    direction = end - start
    direction /= np.linalg.norm(direction)

    # Compute the rotation axis and angle
    up = np.array([0, 0, 1])  # Default up vector of cylinder
    rotation_axis = np.cross(up, direction)
    rotation_angle = np.arccos(np.dot(up, direction))

    # Compute the rotation matrix
    if np.linalg.norm(rotation_axis) != 0:
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        new_cylinder.rotate(rotation_matrix, center=np.array([0, 0, 0]))

    # Translate the new cylinder to the start position
    new_cylinder.translate(start)

    # Copy new cylinder to the original one if it exists
    return new_cylinder


class ReplayDataVisualizer:
    def __init__(self, file):

        self.file = file

        self.K = np.array([
            644.244, 0, 647.406,
            0, 644.244, 354.18,
            0, 0, 1
        ]).reshape(3, 3)

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

        self.calib = calibeyeglove()
        self.startcalib = False
        self.tTg = None

    def reproject(self, rgb_image, depth_image, K):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        H, W, C = rgb_image.shape
        z = depth_image / 1000.
        xi, yi = np.meshgrid(np.arange(W), np.arange(H))
        x = (xi - cx) / fx * z
        y = (yi - cy) / fy * z
        pc = np.stack([x, y, z], axis=-1)

        open3d_cloud = o3d.geometry.PointCloud()
        mask = z < 3.
        open3d_cloud.points = o3d.utility.Vector3dVector(pc[mask, :3])
        open3d_cloud.colors = o3d.utility.Vector3dVector(rgb_image[mask] / 255.)
        # o3d.visualization.draw_geometries([open3d_cloud])
        return open3d_cloud

    def toT(self, msg):
        t = msg[:3]
        q = msg[3:]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(q).as_matrix()
        T[:3, 3] = t
        return T

    def replay(self):
        with h5py.File(self.file, 'r') as root:

            rgb = root["/observations/rgb"]

            settime = 30#root.attrs["settime"]
            depth = root["/observations/depth"]
            compress_length_rgb = root["/compress_len/rgb"]
            compress_length_depth = root["/compress_len/depth"]

            pose_4x4s = root["/coord/base"]

            if "/coord/hand" in root.keys():
                hand_data = root["/coord/hand"]
                #                [data["hand"]["left"][i+1] for i in range(20)] + [data["hand"]["lefttip"][tip] for tip in tips] + \
                # [data["hand"]["right"][i+1] for i in range(20)] + [data["hand"]["righttip"][tip] for tip in tips])
                if hand_data.shape[1] == 350:
                    shift = 0
                else:
                    shift = 23 * 7
                    body = hand_data[:, :shift].reshape(-1, 23, 7)
                left_hand = hand_data[:, shift:shift + 20 * 7].reshape(-1, 20, 7)
                left_tips = hand_data[:, shift + 20 * 7:shift + 25 * 7].reshape(-1, 5, 7)
                right_hand = hand_data[:, shift + 25 * 7:shift + 45 * 7].reshape(-1, 20, 7)
                right_tips = hand_data[:, shift + 45 * 7:shift + 50 * 7].reshape(-1, 5, 7)
            else:
                hand_data = None

            for i in range(len(rgb)):
                pose_4x4 = self.toT(pose_4x4s[i, :7])
                pose_4x4_2 = self.toT(pose_4x4s[i, 7:])

                pose = pose_4x4 @ self.pose  # pose d435 2 t265

                pose2 = pose_4x4_2 @ self.c3

                color_frame = cv2.imdecode(np.asarray(rgb[i, :int(compress_length_rgb[i])]), cv2.IMREAD_COLOR)
                depth_frame = cv2.imdecode(np.asarray(depth[i, :int(compress_length_depth[i])]), cv2.IMREAD_ANYDEPTH)

                depth_image_o3d = o3d.geometry.Image(depth_frame)
                color_image_o3d = o3d.geometry.Image(color_frame[:, :, ::-1].copy())

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_image_o3d,
                    depth_image_o3d,
                    depth_trunc=3.0,
                    convert_rgb_to_intensity=False,
                )
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    1280,
                    720,
                    self.K[0, 0],
                    self.K[1, 1],
                    self.K[0, 2],
                    self.K[1, 2]
                )
                rgbd = o3d.geometry.PointCloud().create_from_rgbd_image(rgbd, intrinsic)

                # rgbd = self.reproject(color_frame,depth_frame,self.K)t

                cv2.imshow("out", color_frame)
                cv2.imshow("depth", cv2.applyColorMap((depth_frame / 1000. * 255).astype(np.uint8), cv2.COLORMAP_JET))
                # cv2.imshow("out1", left_data)
                key = cv2.waitKey(1)

                if i >= settime:
                    self.startcalib = True
                if key == ord('t'):
                    cv2.waitKey()

                self.viser.clear_geometries()
                self.viser.add_geometry(rgbd.transform(pose))

                # meshess = [
                #     o3d.geometry.TriangleMesh().create_sphere(0.01).compute_vertex_normals().transform(self.toT(dd))
                #     for
                #     dd in body[i]]
                # meshess[10].paint_uniform_color([0, 1, 0])
                # [self.viser.add_geometry(m) for m in meshess]

                if hand_data is not None:
                    meshes = []

                    # hand_base = self.toT(body[i][10])  # bTh
                    hand_base2 = self.toT(body[i][4])  # bTw
                    # # base = np.linalg.inv(hand_base)
                    #
                    # for id in range(len(right_hand[i])):
                    #     T = np.eye(4)
                    #     T[:3, 3] = right_hand[i][id][:3]
                    #     T[:3, :3] = Rotation.from_quat(right_hand[i][id][3:]).as_matrix()
                    #     meshes.append(
                    #         o3d.geometry.TriangleMesh().create_sphere(0.005).compute_vertex_normals().transform(T))
                    #     # meshes.append(o3d.geometry.TriangleMesh().create_coordinate_frame(0.01).transform(T))
                    #
                    #     if id in Distal_map.keys():
                    #         part = Distal_map[id]
                    #         start = T[:3, 3]
                    #         end = right_tips[i][tips.index(part)][:3]
                    #         T2 = self.toT(right_tips[i][tips.index(part)])
                    #         meshes.append(create_or_update_cylinder(start, end))
                    #         meshes.append(
                    #             o3d.geometry.TriangleMesh().create_sphere(0.005).compute_vertex_normals().transform(T2))
                    # for (x, y) in lines:
                    #     start = right_hand[i][x][:3]
                    #     end = right_hand[i][y][:3]
                    #     meshes.append(create_or_update_cylinder(start, end))
                    #
                    # for id in range(len(left_hand[i])):
                    #     T = np.eye(4)
                    #     T[:3, 3] = left_hand[i][id][:3]
                    #     T[:3, :3] = Rotation.from_quat(left_hand[i][id][3:]).as_matrix()
                    #     meshes.append(
                    #         o3d.geometry.TriangleMesh().create_sphere(0.005).compute_vertex_normals().transform(T))
                    #     # meshes.append(o3d.geometry.TriangleMesh().create_coordinate_frame(0.01).transform(T))
                    #
                    #     # if id in Distal_map.keys():
                    #     #     part = Distal_map[id]
                    #     #     start = T[:3, 3]
                    #     #     end = left_tips[i][tips.index(part)][:3]
                    #     #     T2 = self.toT(left_tips[i][tips.index(part)])
                    #     #     meshes.append(create_or_update_cylinder(start, end))
                    #     #     meshes.append(
                    #     #         o3d.geometry.TriangleMesh().create_sphere(0.005).compute_vertex_normals().transform(T2))
                    # for (x, y) in lines:
                    #     start = left_hand[i][x][:3]
                    #     end = left_hand[i][y][:3]
                    #     meshes.append(create_or_update_cylinder(start, end))
                    if self.tTg is not None:
                        [self.viser.add_geometry(m.transform(pose_4x4 @ self.tTg @
                                                             np.linalg.inv(hand_base2))) for m in meshes]
                        self.viser.add_geometry(
                            o3d.geometry.TriangleMesh().create_coordinate_frame(0.1).transform(hand_base2 @
                                                                                               np.linalg.inv(self.tTg)))
                        # self.viser.add_geometry(
                        #     o3d.geometry.TriangleMesh().create_coordinate_frame(0.1).transform(pose_4x4))
                    else:
                        [self.viser.add_geometry(m) for m in meshes]
                    # self.viser.add_geometry(
                    #     o3d.geometry.TriangleMesh().create_coordinate_frame(0.3).transform(hand_base))
                    # self.viser.add_geometry(
                    #     o3d.geometry.TriangleMesh().create_coordinate_frame(0.3).transform(hand_base2))

                    if self.startcalib and False:
                        self.calib.insert(hand_base2, pose_4x4)
                        if len(self.calib) == 30:
                            self.tTg = self.calib.calib()
                            self.startcalib = False

                # self.viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1))
                self.viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1).transform(pose_4x4))
                self.viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1).transform(pose2))
                # self.viser.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame(0.1).transform(hand_base2))

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


if __name__ == '__main__':
    viser = ReplayDataVisualizer("test/episode_16.hdf5")
    viser.replay()
