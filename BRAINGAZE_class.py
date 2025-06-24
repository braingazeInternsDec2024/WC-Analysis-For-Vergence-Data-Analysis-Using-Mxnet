# -*- coding: utf-8 -*-
"""
BrainGaze wrapper around gazelazer project.
This class allows for comparison between the vergence extracted from 
tobii x2-30 tracker and a webcam.
@author: Oleksii Leonovych
"""

import os
import numpy as np
from numpy import sin, cos, pi, arctan
from numpy.linalg import norm
import pandas as pd
import base64
from PIL import Image
from io import BytesIO
import cv2
import imageio
from scipy import interpolate, signal
from scipy.signal import savgol_filter, medfilt
import matplotlib.pyplot as plt
from service.head_pose import HeadPoseEstimator
from service.face_alignment import CoordinateAlignmentModel
# Update the import at the top of the file
from service.face_detector import FaceDetectionModel  
from service.iris_localization import IrisLocalizationModel
import time
from queue import Queue
from threading import Thread
from help_functions import *


###############################################################################
class BG:

    def __init__(self, folder_with_log_files):
        self.path_folder = folder_with_log_files
        files = [f for f in os.listdir(self.path_folder) if f.endswith(".mp4")]
        selfiecam_videos = [s for s in files if "_SELFICAM_" in s]
        self.path_selficam_videos = (
            [os.path.join(self.path_folder, video) for video in selfiecam_videos]
            if selfiecam_videos
            else None
        )

        print(self.path_selficam_videos)

    ###############################################################################
    #    this is an adaptation of gazelaser function for input videos
    ###############################################################################

    def vergence_from_selfi_cam_video(self):
        # Initialize DataFrame
        df_vergence_calc = pd.DataFrame(
            columns=[
                "video file name",
                "frame number",
                "timestamp (s)",
                "pupil left radius",
                "pupil right radius",
                "vergence from direction in radians",
                "vergence from direction in degrees",
                "vergence from gaze arrows in radians",
                "vergence from gaze arrows in degrees",
            ]
        )

        # Define DL models to use:
        gpu_ctx = -1
        fd = FaceDetectionModel("weights/16and32", 0, 0.6, gpu=gpu_ctx)
        fa = CoordinateAlignmentModel("weights/2d106det", 0, gpu=gpu_ctx)
        gs = IrisLocalizationModel("weights/iris_landmark.tflite")

        for i, video_path in enumerate(self.path_selficam_videos):
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error: Unable to open video file {video_path}.")
                continue

            # Get width, height, total frames, and fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Define frame size for head pose framework:
            hp = HeadPoseEstimator("weights/object_points.npy", width, height)

            frame_number = 0
            # Process video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                # Print current progress
                print(f"Processing frame {frame_number}/{total_frames}", end="\r")

                # Calculate the timestamp for this frame (starting from 0)
                timestamp = (frame_number - 1) / fps

                bboxes = fd.detect(frame)

                for landmarks in fa.get_landmarks(frame, bboxes, calibrate=True):
                    # Calculate head pose
                    _, euler_angle = hp.get_head_pose(landmarks)
                    pitch, yaw, roll = euler_angle[:, 0]

                    eye_markers = np.take(landmarks, fa.eye_bound, axis=0)
                    eye_centers = np.average(eye_markers, axis=1)
                    eye_lengths = (landmarks[[39, 93]] - landmarks[[35, 89]])[:, 0]

                    # Compute left pupil size and radius
                    iris_left = gs.get_mesh(frame, eye_lengths[0], eye_centers[0])
                    pupil_left, radius_pupil_left = gs.draw_pupil(
                        iris_left, frame, thickness=1
                    )

                    # Compute right pupil size and radius
                    iris_right = gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
                    pupil_right, radius_pupil_right = gs.draw_pupil(
                        iris_right, frame, thickness=1
                    )

                    pupils = np.array([pupil_left, pupil_right])
                    poi = (
                        landmarks[[35, 89]],
                        landmarks[[39, 93]],
                        pupils,
                        eye_centers,
                    )

                    # Calculate vergence
                    delta, angle, (Lx, Ly, Lz), (Rx, Ry, Rz) = calculate_gaze_and_vergence_using_eye_direction(frame, poi)

                    angle_arrows, offset_left, offset_right = process_gaze_data_and_vergence_using_eye_arrows(delta, yaw, roll, pupils)

                    # Prepare a new row of data with the timestamp
                    new_row = {
                        "video file name": os.path.basename(video_path),
                        "frame number": frame_number,
                        "timestamp (s)": timestamp,
                        "pupil left radius": radius_pupil_left,
                        "pupil right radius": radius_pupil_right,
                        "vergence from direction in radians": angle,
                        "vergence from direction in degrees": np.degrees(angle),
                        "vergence from gaze arrows in radians": angle_arrows,
                        "vergence from gaze arrows in degrees": np.degrees(
                            angle_arrows
                        ),
                    }

                    df_vergence_calc = pd.concat(
                        [df_vergence_calc, pd.DataFrame([new_row])], ignore_index=True
                    )

            cap.release()

        # Save this DataFrame to the general folder
        path2save = os.path.join(
            self.path_folder, "Vergence Calculation For Selfi Cam Videos.csv"
        )
        df_vergence_calc.to_csv(path2save, index=False)

        return df_vergence_calc

    def vergence_for_each_selfi_cam_video(self):
        # Define DL models to use:
        gpu_ctx = -1
        fd = MxnetDetectionModel("weights/16and32", 0, 0.6, gpu=gpu_ctx)
        fa = CoordinateAlignmentModel("weights/2d106det", 0, gpu=gpu_ctx)
        gs = IrisLocalizationModel("weights/iris_landmark.tflite")

        for i, video_path in enumerate(self.path_selficam_videos):
            cap = cv2.VideoCapture(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            if not cap.isOpened():
                print(f"Error: Unable to open video file {video_path}.")
                continue

            # Initialize DataFrame for the current video
            df_vergence_calc = pd.DataFrame(
                columns=[
                    "video file name",
                    "frame number",
                    "timestamp (s)",
                    "pupil left radius",
                    "pupil right radius",
                    "vergence from direction in radians",
                    "vergence from direction in degrees",
                    "vergence from gaze arrows in radians",
                    "vergence from gaze arrows in degrees",
                ]
            )

            # Continue with existing code...
            # Get width, height, total frames, and fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Define frame size for head pose framework:
            hp = HeadPoseEstimator("weights/object_points.npy", width, height)

            frame_number = 0
            # Process video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                # Print current progress
                print(f"Processing frame {frame_number}/{total_frames}", end="\r")

                # Calculate the timestamp for this frame (starting from 0)
                timestamp = (frame_number - 1) / fps

                bboxes = fd.detect(frame)

                for landmarks in fa.get_landmarks(frame, bboxes, calibrate=True):
                    # (Continue processing each frame...)

                    # After processing each frame, add data to the DataFrame
                    new_row = {
                        "video file name": video_name,
                        "frame number": frame_number,
                        # Add other fields...
                    }
                    df_vergence_calc = pd.concat(
                        [df_vergence_calc, pd.DataFrame([new_row])], ignore_index=True
                    )

            cap.release()

            # Save this DataFrame to a CSV file named after the video
            csv_filename = f"{video_name}_Vergence_Calculation.csv"
            path2save = os.path.join(self.path_folder, csv_filename)
            df_vergence_calc.to_csv(path2save, index=False)

            print(f"Data saved for video {video_name}")

        return df_vergence_calc

    ###############################################################################

    ### PLOTTING ###

    ###############################################################################
    # 1) CREATE THE AVERGAGE PLOT
