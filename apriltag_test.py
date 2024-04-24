
import pyrealsense2 as rs
import pupil_apriltags as apriltag
import numpy as np
import open3d as o3d
import cv2
import pickle as pkl

from transforms3d.quaternions import axangle2quat, qmult, quat2mat, mat2quat



class RealsesneProcessor:
    def __init__(
            self,
            first_t265_serial
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
        self.pipeline = rs.pipeline()
        config = rs.config()

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

        self.vis = None

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


        # get pose data for t265 1
        pose_4x4 = RealsesneProcessor.frame_to_pose_conversion(
            input_t265_frames=t265_frames
        )
        left = t265_frames.get_fisheye_frame(1)
        left_data = np.asanyarray(left.get_data())



        pose = pose_4x4 @ self.pose  # pose d435 2 t265


        cv2.imshow("out1", left_data)
        key = cv2.waitKey(1)

        return key

    def save(self):
        with open("1.pkl", 'wb') as f:
            pkl.dump({
                'd435': self.d435_frame,
                't265': self.t265_pose
            }, f)




# Initialize the realsense camera
realsense_processor = RealsesneProcessor(first_t265_serial="943222111176")
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# pipeline.start(config)

realsense_processor.configure_stream()
# Initialize the apriltag detector
at_detector = apriltag.Detector()

try:
    while True:
        # Get frames from the realsense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert the color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Detect apriltags in the color image
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        detections = at_detector.detect(gray)

        for detection in detections:
            # Calculate the pose of the apriltag
            pose = detection.homography

            # Print the pose of the apriltag
            print("Apriltag pose: \n", pose)

        # Display the color image with apriltag detections
        cv2.imshow('Apriltag Detection', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the realsense pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
