import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.nn import DataParallel
from models.eyenet import EyeNet
from datetime import datetime
import dlib
from imutils import face_utils
from util.eye_prediction import EyePrediction
from util.eye_sample import EyeSample
from typing import List, Optional

from help_functions import *

from service.head_pose import HeadPoseEstimator
from service.face_alignment import CoordinateAlignmentModel
from service.face_detector import MxnetDetectionModel
from service.iris_localization import IrisLocalizationModel

import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize device
torch.backends.cudnn.enabled = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load models and assets
face_cascade = cv2.CascadeClassifier(os.path.join(SCRIPT_DIR, "lbpcascade_frontalface_improved.xml"))
landmarks_detector = dlib.shape_predictor(os.path.join(SCRIPT_DIR, "shape_predictor_68_face_landmarks.dat"))
checkpoint = torch.load(os.path.join(SCRIPT_DIR, "checkpoint.pt"), map_location=device)
eyenet = EyeNet(
    nstack=checkpoint["nstack"],
    nfeatures=checkpoint["nfeatures"],
    nlandmarks=checkpoint["nlandmarks"],
).to(device)
eyenet.load_state_dict(checkpoint["model_state_dict"])

# Define 3D model points of a standard human face
model_points = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float32,
)

# Distortion coefficients (Assume zero for simplicity)
dist_coeffs = np.zeros((4, 1))


class VergenceCalculator:
    def __init__(self, data_directory, output_directory):
        self.data_directory = data_directory
        self.output_directory = output_directory

        # Ensure the output directory exists
        os.makedirs(self.output_directory, exist_ok=True)

    def process_videos(self):
        print("Starting video processing...")
        gpu_ctx = -1
        fd = MxnetDetectionModel(os.path.join(SCRIPT_DIR, "weights/16and32"), 0, 0.6, gpu=gpu_ctx)
        fa = CoordinateAlignmentModel(os.path.join(SCRIPT_DIR, "weights/2d106det"), 0, gpu=gpu_ctx)
        gs = IrisLocalizationModel(os.path.join(SCRIPT_DIR, "weights/iris_landmark.tflite"))

        cascade_path = os.path.join(SCRIPT_DIR, "haarcascade_eye.xml")
        eye_haarcascade = cv2.CascadeClassifier(cascade_path)

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

        # Find all MP4 files recursively
        video_files = []
        for root, dirs, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)
                    parent_folder = os.path.basename(os.path.dirname(video_path))
                    video_files.append((video_path, parent_folder))
        
        print(f"Found {len(video_files)} MP4 files in total")
        
        # Process each video
        for video_path, parent_folder in video_files:
            # Extract video name without extension
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Create CSV filename with both parent folder and video name
            csv_filename = os.path.join(
                self.output_directory,
                f"Vergence_Calculation_{video_name}.csv",
            )

            print(f"Processing video: {video_path}")
            print(f"Output will be saved to: {csv_filename}")
            
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
            
            # Continue with the rest of the video processing code...

            for folder in os.listdir(self.data_directory):
                folder_path = os.path.join(self.data_directory, folder)
            
                if os.path.isdir(folder_path):  
                    for f in os.listdir(folder_path):
                        video_path = ''
                        if f.endswith(".mp4"):  
                            video_path = os.path.join(folder_path, f)

                        csv_filename = os.path.join(
                            self.output_directory,
                            f"Vergence_Combined_Calculation_of_{os.path.basename(folder_path)}.csv",
                        )


                        print(f"Processing video: {video_path}")
                        print(f"Processing video: {video_path}")

                        # Get rotation metadata
                        rotation_angle = self.get_video_rotation(video_path)
                        print(f"Rotation metadata: {rotation_angle} degrees")

                        cap = cv2.VideoCapture(video_path)
                        video_name = os.path.splitext(os.path.basename(video_path))[0]

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
                        hp = HeadPoseEstimator(os.path.join(SCRIPT_DIR,"weights/object_points.npy"), width, height)

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

                            # Define checkerboard dimensions
                            checkerboard_size = (6, 9)
                            square_size = 25  # mm

                            # Prepare object points
                            objp = np.zeros(
                                (checkerboard_size[0] * checkerboard_size[1], 3), np.float32
                            )
                            objp[:, :2] = np.mgrid[
                                0 : checkerboard_size[0], 0 : checkerboard_size[1]
                            ].T.reshape(-1, 2)
                            objp *= square_size

                            # Detect corners
                            ret2, corners = cv2.findChessboardCorners(
                                gray_image,
                                checkerboard_size,
                                cv2.CALIB_CB_ADAPTIVE_THRESH
                                + cv2.CALIB_CB_NORMALIZE_IMAGE
                                + cv2.CALIB_CB_FAST_CHECK,
                            )

                            focal_length = width

                            if ret2:
                                _, mtx, dist, _, _ = cv2.calibrateCamera(
                                    [objp], [corners], gray_image.shape[::-1], None, None
                                )
                                focal_length = mtx[0, 0]  # Extract focal length in pixels
                                print(f"Focal length: {focal_length}")

                            center = (width / 2, height / 2)
                            camera_matrix = np.array(
                                [
                                    [focal_length, 0, center[0]],
                                    [0, focal_length, center[1]],
                                    [0, 0, 1],
                                ],
                                dtype=np.float32,
                            )

                            # Using first model
                            # bboxes = fd.detect(frame)
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

                            # Using second model (EyeNet)
                            orig_frame = frame.copy()

                            leftmost_points = [
                                landmarks[35],
                                landmarks[89],
                            ]

                            rightmost_points = [landmarks[39], landmarks[93]]

                            # Annotate the POI on the frame
                            self.annotate_pois(
                                orig_frame,
                                leftmost_points,
                                rightmost_points,
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

                            df_vergence_calc = pd.concat(
                                [df_vergence_calc, pd.DataFrame([new_row])], ignore_index=True
                            )

                            # cv2.imshow("Processed Frame", orig_frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break

                        cap.release()
                        print(f"Finished processing video: {video_path}")

                        df_vergence_calc.to_csv(csv_filename, index=False)
            # print(f"Combined data saved to {csv_filename}")

    def draw_cascade_face(self, face, frame):
        (x, y, w, h) = (int(e) for e in face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def draw_colored_landmarks(self, landmarks, frame):
        # Landmark indices:
        # 0,1 = Right eye corners
        # 2,3 = Left eye corners
        # 4 = Nose tip

        # Define colors in BGR
        right_eye_color = (0, 0, 255)  # Red for right eye
        left_eye_color = (255, 0, 0)  # Blue for left eye
        nose_color = (0, 255, 0)  # Green for nose

        # Draw right eye landmarks (0,1)
        for i in [0, 1]:
            x, y = landmarks[i]
            cv2.circle(
                frame,
                (int(x), int(y)),
                2,
                right_eye_color,
                -1,
                lineType=cv2.LINE_AA,
            )

        # Draw left eye landmarks (2,3)
        for i in [2, 3]:
            x, y = landmarks[i]
            cv2.circle(
                frame, (int(x), int(y)), 2, left_eye_color, -1, lineType=cv2.LINE_AA
            )

        # Draw nose landmark (4)
        x, y = landmarks[4]
        cv2.circle(frame, (int(x), int(y)), 2, nose_color, -1, lineType=cv2.LINE_AA)

    def get_eye_points(self, landmarks, indices):
        return [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]

    def detect_pupil(self, frame, eye_points):
        x_coordinates, y_coordinates = zip(*eye_points)
        x_min, x_max = min(x_coordinates), max(x_coordinates)
        y_min, y_max = min(y_coordinates), max(y_coordinates)

        # Add padding to the bounding box
        padding = 2
        x_min = max(0, x_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(frame.shape[0], y_max + padding)

        # Crop the eye region
        eye_region = frame[y_min:y_max, x_min:x_max]

        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # Apply adaptive thresholding
        thresholded = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours and select the largest one
        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
            return radius
        return 0

    def estimate_pupil_radius(self, landmarks, eye_indices):
        # Extract the eye landmarks
        eye_points = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices]
        )

        # Calculate the bounding box of the eye
        x_coordinates, y_coordinates = zip(*eye_points)
        min_x, max_x = min(x_coordinates), max(x_coordinates)
        min_y, max_y = min(y_coordinates), max(y_coordinates)

        # Approximate the pupil radius
        eye_width = max_x - min_x
        eye_height = max_y - min_y
        pupil_radius = (
            min(eye_width, eye_height) * 0.25
        )  # Estimate: 50% of the iris diameter, iris approximated by min dimension

        return pupil_radius

    def find_eye_centers(self, landmarks):
        """Approximate the eye centers using the average of eye contour landmarks."""

        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)

        return np.array([left_eye_center, right_eye_center])

    def annotate_pois(
        self, frame, leftmost_points, rightmost_points, pupils, eye_centers
    ):
        """
        Annotate Points of Interest (POIs) on the frame using coordinates from dlib points.
        """

        # Colors for different annotations
        color_leftmost = (255, 0, 0)  # Red for leftmost points
        color_rightmost = (0, 255, 0)  # Green for rightmost points
        color_pupil = (0, 0, 255)  # Blue for pupils
        color_center = (255, 255, 0)  # Yellow for eye centers

        # circle radius
        radius_leftmost = 3  # Smaller radius for leftmost points
        radius_rightmost = 3  # Smaller radius for rightmost points
        radius_pupil = 4  # Slightly larger radius for pupils, but smaller than before
        radius_center = 4  # Smaller radius for eye centers

        # Annotate leftmost points
        for idx, point in enumerate(leftmost_points):
            cv2.circle(
                frame,
                (int(point[0]), int(point[1])),
                radius_leftmost,
                color_leftmost,
                -1,
            )
            cv2.putText(
                frame,
                f"L{idx + 1}",
                (int(point[0]) + 5, int(point[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_leftmost,
                1,
            )

        # Annotate rightmost points
        for idx, point in enumerate(rightmost_points):
            cv2.circle(
                frame,
                (int(point[0]), int(point[1])),
                radius_rightmost,
                color_rightmost,
                -1,
            )
            cv2.putText(
                frame,
                f"R{idx + 1}",
                (int(point[0]) + 5, int(point[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_rightmost,
                1,
            )

        # Annotate pupils
        for idx, pupil in enumerate(pupils):
            cv2.circle(
                frame, (int(pupil[0]), int(pupil[1])), radius_pupil, color_pupil, -1
            )
            cv2.putText(
                frame,
                f"P{idx + 1}",
                (int(pupil[0]) + 5, int(pupil[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_pupil,
                1,
            )

        # Annotate eye centers
        for idx, center in enumerate(eye_centers):
            cv2.circle(
                frame, (int(center[0]), int(center[1])), radius_center, color_center, -1
            )
            cv2.putText(
                frame,
                f"C{idx + 1}",
                (int(center[0]) + 5, int(center[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_center,
                1,
            )

    def segment_eyes(self, frame, landmarks, ow=160, oh=96):
        eyes = []

        # Convert dlib landmarks to a NumPy array
        landmarks_array = np.array(
            [[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)]
        )

        # Segment eyes
        for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
            x1, y1 = landmarks_array[corner1, :]
            x2, y2 = landmarks_array[corner2, :]
            eye_width = 1.5 * np.linalg.norm(
                landmarks_array[corner1, :] - landmarks_array[corner2, :]
            )
            if eye_width == 0.0:
                return eyes

            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

            # Center image on middle of eye
            translate_mat = np.asmatrix(np.eye(3))
            translate_mat[:2, 2] = [[-cx], [-cy]]
            inv_translate_mat = np.asmatrix(np.eye(3))
            inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

            # Scale
            scale = ow / eye_width
            scale_mat = np.asmatrix(np.eye(3))
            scale_mat[0, 0] = scale_mat[1, 1] = scale
            inv_scale = 1.0 / scale
            inv_scale_mat = np.asmatrix(np.eye(3))
            inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

            estimated_radius = 0.5 * eye_width * scale

            # Center image
            center_mat = np.asmatrix(np.eye(3))
            center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
            inv_center_mat = np.asmatrix(np.eye(3))
            inv_center_mat[:2, 2] = -center_mat[:2, 2]

            # Get rotated and scaled, and segmented image
            transform_mat = center_mat * scale_mat * translate_mat
            inv_transform_mat = inv_translate_mat * inv_scale_mat * inv_center_mat

            eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
            eye_image = cv2.equalizeHist(eye_image)

            if is_left:
                eye_image = np.fliplr(eye_image)
                cv2.imshow("left eye image", eye_image)
            else:
                cv2.imshow("right eye image", eye_image)

            eyes.append(
                EyeSample(
                    orig_img=frame.copy(),
                    img=eye_image,
                    transform_inv=inv_transform_mat,
                    is_left=is_left,
                    estimated_radius=estimated_radius,
                )
            )
        return eyes

    def smooth_eye_landmarks(
        self,
        eye: EyePrediction,
        prev_eye: Optional[EyePrediction],
        smoothing=0.2,
        gaze_smoothing=0.4,
    ):
        if prev_eye is None:
            return eye
        return EyePrediction(
            eye_sample=eye.eye_sample,
            landmarks=smoothing * prev_eye.landmarks + (1 - smoothing) * eye.landmarks,
            gaze=gaze_smoothing * prev_eye.gaze + (1 - gaze_smoothing) * eye.gaze,
        )

    def run_eyenet(self, eyes: List[EyeSample], ow=160, oh=96) -> List[EyePrediction]:
        print("Running EyeNet...")
        result = []
        for eye in eyes:
            with torch.no_grad():
                x = torch.tensor([eye.img], dtype=torch.float32).to(device)
                _, landmarks, gaze = eyenet.forward(x)
                landmarks = np.asarray(landmarks.cpu().numpy()[0])
                gaze = np.asarray(gaze.cpu().numpy()[0])
                assert gaze.shape == (2,)
                assert landmarks.shape == (34, 2)

                landmarks = landmarks * np.array([oh / 48, ow / 80])

                temp = np.zeros((34, 3))
                if eye.is_left:
                    temp[:, 0] = ow - landmarks[:, 1]
                else:
                    temp[:, 0] = landmarks[:, 1]
                temp[:, 1] = landmarks[:, 0]
                temp[:, 2] = 1.0
                landmarks = temp
                assert landmarks.shape == (34, 3)
                landmarks = np.asarray(np.matmul(landmarks, eye.transform_inv.T))[:, :2]
                assert landmarks.shape == (34, 2)
                result.append(
                    EyePrediction(eye_sample=eye, landmarks=landmarks, gaze=gaze)
                )
        return result

    def spherical_to_cartesian(self, pitch, yaw):
        vx = np.cos(pitch) * np.sin(yaw)
        vy = np.sin(pitch)
        vz = np.cos(pitch) * np.cos(yaw)
        v = np.array([vx, vy, vz])
        v = v / np.linalg.norm(v)  # Normalize to get a unit vector
        return v

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
