import time

import numpy as np


from pyorbbecsdk import *
from camera_utils import frame_to_bgr_image


class GraspCameraORB:
    def __init__(self):
        super().__init__()
        self.soft = False
        self.K = np.array([688.6412353515625, 0., 644.2440185546875,
                       0., 688.4562377929688, 360.9799499511719,
                       0., 0., 1.],
                      np.float32).reshape(3, 3)
        self.open()


    def open(self):
        pipeline = Pipeline()
        config = Config()
        ctx = Context()
        ctx.set_logger_to_console(OBLogLevel.ERROR)

        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

        if depth_profile_list is None:
            print("No proper depth profile, can not generate point cloud")
            return
        depth_profile:VideoStreamProfile = depth_profile_list.get_default_video_stream_profile()

        config.enable_stream(depth_profile)

        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is not None:
                color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
                config.enable_stream(color_profile)
                config.set_align_mode(OBAlignMode.HW_MODE)
                pipeline.enable_frame_sync()

        except OBError as e:
            config.set_align_mode(OBAlignMode.DISABLE)
            print(e)
        self.device = pipeline.get_device()
        self.device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_SOFT_FILTER_BOOL, self.soft)

        pipeline.start(config)

        """
        [[691.949     0.      645.53406]
 [  0.      691.66077 356.00116]
 [  0.        0.        1.     ]]
        """
        self.pipeline = pipeline
        cc = pipeline.get_camera_param()
        self.K[0,0] = cc.depth_intrinsic.fx
        self.K[1,1] = cc.depth_intrinsic.fy
        self.K[0,2] = cc.depth_intrinsic.cx
        self.K[1,2] = cc.depth_intrinsic.cy
        print(self.K)


    def read(self):
        try:
            frames = None
            while frames is None:
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if depth_frame is None or color_frame is None:
                    frames = None

            width = depth_frame.get_width()
            height = depth_frame.get_height()

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)

            depth_data = depth_data.reshape((height, width))

            color = frame_to_bgr_image(color_frame)

            return True, (color, depth_data)

        except OBError as e:
            print(e)
            exit(0)

    def set_soft(self):
        self.soft = not self.soft
        self.device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_SOFT_FILTER_BOOL, self.soft)

if __name__ == '__main__':
    import cv2
    cap = GraspCameraORB()
    while True:
        flag,frame = cap.read()
        if not flag:
            continue
        cv2.imshow("out",frame[0])
        cv2.waitKey(1)