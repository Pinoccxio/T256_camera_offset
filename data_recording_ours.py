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


def get_auto_index(dataset_dir, dataset_name_prefix='', data_suffix='hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def main(args):
    realsense_processor = RealsenseProcessor(
        first_t265_serial="146322110119",
        second_t265_serial="146322110372",
        third_t265_serial="929122111181",
        save_hand=args.save_hand
    )
    exit_program = False

    def signal_handler(signal, frame):
        exit_program = True

    signal.signal(signal.SIGINT, signal_handler)

    data_dict = {
        '/observations/rgb': [],
        '/observations/depth': [],
        '/coord/base': [],  # t265_1 t265_2 2 * 7
    }
    if args.save_hand:
        data_dict.update({'/coord/hand': []})  # (20+5) * 2 * 7
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    encode_rgb_length_max = 0
    encode_rgb_length = []
    encode_depth_length_max = 0
    encode_depth_length = []
    max_length = 10000

    realsense_processor.configure_stream()
    tips = [
        "thumb",
        "index",
        "middle",
        "ring",
        "little"]
    length = 0

    dataset_path = args.log_dir
    os.makedirs(dataset_path, exist_ok=True)
    filename = f"episode_{get_auto_index(dataset_path)}.hdf5"

    FPS = 20
    time0 = time.time()
    DT = 1 / FPS
    settime = -1

    # try:
    while not exit_program and length < max_length:
        t0 = time.time()
        data = realsense_processor.process_frame()
        rgb, depth = data["d435"]

        result, encoded_image = cv2.imencode('.jpg', rgb)
        data_dict["/observations/rgb"].append(encoded_image)
        encode_rgb_length.append(len(encoded_image))
        if len(encoded_image) > encode_rgb_length_max:
            encode_rgb_length_max = len(encoded_image)

        result, encoded_image = cv2.imencode('.png', depth)
        data_dict["/observations/depth"].append(encoded_image)
        if len(encoded_image) > encode_depth_length_max:
            encode_depth_length_max = len(encoded_image)
        encode_depth_length.append(len(encoded_image))

        if args.save_hand:
            hand_params = np.concatenate(
                [data["hand"]['body'][i + 1] for i in range(23)] +
                [data["hand"]["left"][i + 1] for i in range(20)] + [data["hand"]["lefttip"][tip] for tip in tips] + \
                [data["hand"]["right"][i + 1] for i in range(20)] + [data["hand"]["righttip"][tip] for tip in tips])
            data_dict["/coord/hand"].append(hand_params)

        data_dict["/coord/base"].append(np.concatenate([data["t2651"], data["t2652"]]))

        cv2.imshow("out", rgb)
        cv2.imshow("depth", cv2.applyColorMap((depth / 1000. * 255).astype(np.uint8), cv2.COLORMAP_JET))
        key = cv2.waitKey(1)

        length += 1

        time.sleep(max(0, DT - (time.time() - t0)))

        if key == ord('q'):
            break
        elif key == ord('e'):
            settime = length

    # except Exception as e:
    #    print_exc(e)
    # finally:
    duration = time.time() - time0
    print(f'Avg fps: {length / (duration):.3f}({length:d}/{duration:.3f}s)')
    realsense_processor.close()
    # resolve compress
    print("Saveing")
    padded_compressed_image_list = []
    for compressed_image in data_dict["/observations/rgb"]:
        padded_compressed_image = np.zeros(encode_rgb_length_max, dtype='uint8')
        image_len = len(compressed_image)
        padded_compressed_image[:image_len] = compressed_image
        padded_compressed_image_list.append(padded_compressed_image)
    data_dict[f'/observations/rgb'] = padded_compressed_image_list

    padded_compressed_image_list = []
    for compressed_image in data_dict["/observations/depth"]:
        padded_compressed_image = np.zeros(encode_depth_length_max, dtype='uint8')
        image_len = len(compressed_image)
        padded_compressed_image[:image_len] = compressed_image
        padded_compressed_image_list.append(padded_compressed_image)
    data_dict[f'/observations/depth'] = padded_compressed_image_list

    with h5py.File(os.path.join(dataset_path, filename), 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = True
        root.attrs['settime'] = settime
        root.attrs["time"] = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        obs = root.create_group('coord')
        image = root.create_group('observations')

        _ = image.create_dataset("rgb", (length, encode_rgb_length_max), dtype='uint8',
                                 chunks=(1, encode_rgb_length_max), )
        _ = image.create_dataset("depth", (length, encode_depth_length_max), dtype='uint8',
                                 chunks=(1, encode_depth_length_max), )
        if args.save_hand:
            _ = obs.create_dataset('hand', (length, 23 * 7 + 25 * 2 * 7))
        _ = obs.create_dataset("base", (length, 2 * 7))

        for name, array in data_dict.items():
            root[name][...] = array

        comlen = root.create_group('compress_len')
        _ = comlen.create_dataset('rgb', (length))
        _ = comlen.create_dataset('depth', (length))
        root["/compress_len/rgb"][...] = encode_rgb_length
        root["/compress_len/depth"][...] = encode_depth_length
    print("Saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hand_rec")

    parser.add_argument("--log_dir", default="test", type=str)
    parser.add_argument("--save_hand", action="store_true", default=True)

    args = parser.parse_args()

    main(args)
