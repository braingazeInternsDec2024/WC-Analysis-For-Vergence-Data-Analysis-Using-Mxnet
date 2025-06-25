import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import subprocess

from help_functions import *

from service.head_pose import HeadPoseEstimator
from service.face_alignment import CoordinateAlignmentModel
from service.face_detector import MxnetDetectionModel
from service.iris_localization import IrisLocalizationModel


class VergenceCalculator:
    def __init__(self, data_directory, output_directory):
        self.data_directory = data_directory
        self.output_directory = output_directory

        # Ensure the output directory exists
        os.makedirs(self.output_directory, exist_ok=True)

    def process_videos(self):
        print("Starting video processing...")
        gpu_ctx = -1
        fd = MxnetDetectionModel("weights/16and32", 0, 0.6, gpu=gpu_ctx)
        fa = CoordinateAlignmentModel("weights/2d106det", 0, gpu=gpu_ctx)
        gs = IrisLocalizationModel("weights/iris_landmark.tflite")

        df_vergence_calc = pd.DataFrame(
            columns=[
                "video file name",
                "frame number",
                "timestamp (s)",
                "ML model",
                "pupil left radius",
                "pupil right radius",
                "vergence from direction in radians",
                "vergence from direction in degrees",
                "vergence from gaze arrows in radians",
                "vergence from gaze arrows in degrees",
            ]
        )

        # Create a list of all video files in the directory and its subdirectories
        videos_to_process = []
        for root, dirs, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith(".mp4"):
                    videos_to_process.append(os.path.join(root, file))

        for video_path in videos_to_process:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            csv_filename = os.path.join(
                self.output_directory,
                f"Vergence_{video_name}.csv",
            )

            print(f"Processing video: {video_path}")

            # Get rotation metadata
            rotation_angle = self.get_video_rotation(video_path)
            print(f"Rotation metadata: {rotation_angle} degrees")

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error: Unable to open video file {video_path}.")
                continue

            frame_number = 0

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(
                f"Video properties: Width={width}, Height={height}, Total Frames={total_frames}, FPS={fps}"
            )

            # Define frame size for head pose framework:
            hp = HeadPoseEstimator("weights/object_points.npy", width, height)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached.")
                    break

                # Rotate frame based on metadata
                if rotation_angle == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation_angle == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation_angle == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
                gray_image = cv2.equalizeHist(gray_image)

                frame_number += 1
                timestamp = (frame_number - 1) / fps
                print(f"Processing frame {frame_number}, Timestamp: {timestamp:.2f}s")

                # Using WC Analysis model
                bboxes = list(fd.detect(frame))
                print(f"Detected {len(bboxes)} faces.")

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
                    delta, angle, (Lx, Ly, Lz), (Rx, Ry, Rz) = (
                        calculate_gaze_and_vergence_using_eye_direction(frame, poi)
                    )

                    angle_arrows, offset_left, offset_right = (
                        process_gaze_data_and_vergence_using_eye_arrows(
                            delta, yaw, roll, pupils
                        )
                    )

                    # Prepare a new row of data with the timestamp
                    new_row = {
                        "video file name": os.path.basename(video_path),
                        "frame number": frame_number,
                        "timestamp (s)": timestamp,
                        "ML model": "WC Analysis",
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
            print(f"Finished processing video: {video_path}")

            df_vergence_calc.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")

    def get_video_rotation(self, video_path):
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream_tags=rotate",
                    "-of",
                    "csv=p=0",
                    video_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return int(result.stdout.strip()) if result.stdout.strip() else 0
        except Exception as e:
            print(f"Error getting rotation metadata for {video_path}: {e}")
            return 0
