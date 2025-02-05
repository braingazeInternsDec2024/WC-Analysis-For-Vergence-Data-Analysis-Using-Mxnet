# -*- coding: utf-8 -*-
"""
help functions for gatekeeper project
@author: leono
"""
# import os
import numpy as np
import pandas as pd
import base64
import math
from PIL import Image
from io import BytesIO
import cv2
import imageio
from scipy import interpolate, signal
from scipy.signal import savgol_filter, medfilt
from numpy import sin, cos, pi, arctan
from numpy.linalg import norm
import matplotlib.pyplot as plt


# mysterious definitions:
SIN_LEFT_THETA = 2 * sin(pi / 4)  # 1.41
SIN_UP_THETA = sin(pi / 6)  # 0.5


def interp(x, y, uni_grid):
    # x = list(trial.TimeStamp)
    # y = list(trial['vergence from tobii in degrees'])
    f = interpolate.interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
    xnew = uni_grid
    ynew = f(xnew)
    return ynew


# def txt_to_csv(path2log):
#     # read txt:
#     f = open(path2log, "r")
#     log = f.readlines()

#     # find framerate
#     fps_ind = [i for i, s in enumerate(log) if "FrameRate" in s][0]
#     fps = int("".join(filter(str.isdigit, log[fps_ind])))

#     # find index of a line where the tracking data starts
#     ind = [i for i, s in enumerate(log) if "TimeStamp;Image;Trial;State" in s][0]

#     # cut the log file:
#     log = log[ind + 1 :]
#     df = pd.DataFrame(columns=["TimeStamp", "Image", "Trial", "State"])
#     for i in range(len(log)):
#         line = str(log[i]).split(";")
#         if len(line) != 4:
#             continue
#         df.loc[len(df)] = line
#     # save dataframe
#     df.to_csv(path2log[:-3] + "csv")
#     return df, fps


def txt_to_csv(path2log):
    import pandas as pd

    # Read txt file
    with open(path2log, "r") as f:
        log = f.readlines()

    # Find framerate
    fps_ind = [i for i, s in enumerate(log) if "FrameRate" in s][0]
    fps = int("".join(filter(str.isdigit, log[fps_ind])))

    # Find the index where the tracking data starts
    ind = [i for i, s in enumerate(log) if "TimeStamp;Image;Trial;State" in s][0]

    # Cut the log file
    log = log[ind + 1 :]
    df = pd.DataFrame(columns=["TimeStamp", "Image", "Trial", "State"])
    for i in range(len(log)):
        line = str(log[i]).strip().split(";")
        if len(line) != 4:
            continue
        df.loc[len(df)] = line

    # Save dataframe without an index column
    df.to_csv(path2log[:-3] + "csv", index=False)

    return df, fps


def base64_to_image(string):
    im = Image.open(BytesIO(base64.b64decode(string)))
    return im


def readb64(string):
    nparr = np.fromstring(base64.b64decode(string), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def create_video_from_log(df, fps, path_4_video):
    total_frames = len(df)
    write_to = path_4_video
    writer = imageio.get_writer(write_to, format="mp4", mode="I", fps=fps)
    for i in range(total_frames):
        im = base64_to_image(df.iloc[i, 1])
        writer.append_data(np.asarray(im))
    writer.close()


# def process_tobii_log(path2tobii):
#     with open(path2tobii, "r") as f:
#         log = f.readlines()

#     # Find index of a line where the tracking data starts
#     indices = [i for i, s in enumerate(log) if "---" in s]
#     if not indices:
#         print("No delimiter '---' found in the log file.")
#         return pd.DataFrame()  # Return an empty DataFrame if no delimiter is found

#     ind = indices[0]

#     # Cut the log file:
#     log = log[ind + 2 :]

#     # Create a DataFrame to store all the data:
#     columns = [
#         "TimeStamp",
#         "Trackerstamp",
#         "leftEyeRawX",
#         "leftEyeRawY",
#         "rightEyeRawX",
#         "rightEyeRawY",
#         "leftEyePupilSize",
#         "rightEyePupilSize",
#         "leftEyeValidity",
#         "rightEyeValidity",
#         "leftEye3dX",
#         "leftEye3dY",
#         "leftEye3dZ",
#         "rightEye3dX",
#         "rightEye3dY",
#         "rightEye3dZ",
#         "validityScore",
#         "leftGazePoint3dX",
#         "leftGazePoint3dY",
#         "leftGazePoint3dZ",
#         "rightGazePoint3dX",
#         "rightGazePoint3dY",
#         "rightGazePoint3dZ",
#         "Trial",
#         "State",
#     ]
#     df = pd.DataFrame(columns=columns)

#     # Read log line by line
#     for line in log:
#         split_line = line.strip().split(";")
#         # Debugging: Print the split line and its length
#         print("Split line:", split_line)
#         print("Length of split line:", len(split_line))
#         print("Expected columns length:", len(columns))
#         if len(split_line) != len(columns):
#             print("Skipping line due to incorrect number of columns")
#             continue
#         # Explicitly assigning values to DataFrame columns
#         df = df.append(pd.Series(split_line, index=columns), ignore_index=True)

#     return df


def process_tobii_log(path2tobii):
    f = open(path2tobii, "r")
    log = f.readlines()

    # find index of a line where the tracking data starts
    ind = [i for i, s in enumerate(log) if "---" in s][0]

    # cut the log file:
    log = log[ind + 2 :]

    # create a dataframe to store all the data:
    df = pd.DataFrame(
        columns=[
            "TimeStamp",
            "Trackerstamp",
            "leftEyeRawX",
            "leftEyeRawY",
            "rightEyeRawX",
            "rightEyeRawY",
            "leftEyePupilSize",
            "rightEyePupilSize",
            "leftEyeValidity",
            "rightEyeValidity",
            "leftEye3dX",
            "leftEye3dY",
            "leftEye3dZ",
            "rightEye3dX",
            "rightEye3dY",
            "rightEye3dZ",
            "validityScore",
            "leftGazePoint3dX",
            "leftGazePoint3dY",
            "leftGazePoint3dZ",
            "rightGazePoint3dX",
            "rightGazePoint3dY",
            "rightGazePoint3dZ",
            "Trial",
            "State",
        ]
    )

    # read log line by line
    for i in range(len(log)):
        line = str(log[i]).split(";")
        if len(line) != 25:
            continue
        df.loc[len(df)] = line
    return df


def filter_validity(df):
    # discard the rows of a dataframe where validity is not equal to 0 for both left and right eye
    left_val = df[df["leftEyeValidity"] == "0"].index.values
    right_val = df[df["rightEyeValidity"] == "0"].index.values

    list_as_set = set(left_val)
    intersection = list_as_set.intersection(right_val)
    ind_df = list(intersection)
    df = df.iloc[ind_df]
    df.index = np.arange(len(df))
    return df


def ser_2float(series):
    out = series.astype(str)
    # replace commas with points
    out = [w.replace(",", ".") for w in out]
    out = [w.replace("None", "nan") for w in out]
    # make instances a "float value"
    out = pd.Series([float(i) for i in out])
    # interpolate inner gaps
    out = out.interpolate(method="slinear")
    # interp start and end
    out = out.interpolate(method="linear", limit_direction="both")
    # Smooth the signal using  :
    # Savitzky–Golay filter
    # out_smoothed = pd.Series( savgol_filter(out, 7, 1) )
    # Median filter
    out_smoothed = pd.Series(medfilt(out, kernel_size=9))
    return out_smoothed


def gaze_from_4_points(df):
    ##define 4 points in 3D space (2 for eyes and 2 for gaze locations)
    left_gaze_x = ser_2float(df.loc[:, "leftGazePoint3dX"])
    left_gaze_y = ser_2float(df.loc[:, "leftGazePoint3dY"])
    left_gaze_z = ser_2float(df.loc[:, "leftGazePoint3dZ"])

    right_gaze_x = ser_2float(df.loc[:, "rightGazePoint3dX"])
    right_gaze_y = ser_2float(df.loc[:, "rightGazePoint3dY"])
    right_gaze_z = ser_2float(df.loc[:, "rightGazePoint3dZ"])

    left_eye_x = ser_2float(df.loc[:, "leftEye3dX"])
    left_eye_y = ser_2float(df.loc[:, "leftEye3dY"])
    left_eye_z = ser_2float(df.loc[:, "leftEye3dZ"])

    right_eye_x = ser_2float(df.loc[:, "rightEye3dX"])
    right_eye_y = ser_2float(df.loc[:, "rightEye3dY"])
    right_eye_z = ser_2float(df.loc[:, "rightEye3dZ"])
    # vectors originating at gaze pos and ending at pupil coordinates
    gaze_vector_l = [
        list(a)
        for a in zip(
            list(left_eye_x - left_gaze_x),
            list(left_eye_y - left_gaze_y),
            list(left_eye_z - left_gaze_z),
        )
    ]
    gaze_vector_r = [
        list(a)
        for a in zip(
            list(right_eye_x - right_gaze_x),
            list(right_eye_y - right_gaze_y),
            list(right_eye_z - right_gaze_z),
        )
    ]
    # angle between 2 vectors
    angle = []
    for i in range(len(gaze_vector_r)):
        angle.append(
            np.arccos(
                np.dot(gaze_vector_l[i], gaze_vector_r[i])
                / (np.linalg.norm(gaze_vector_l[i]) * np.linalg.norm(gaze_vector_r[i]))
            )
        )

    return angle, gaze_vector_l, gaze_vector_r


def calculate_gaze_and_vergence_using_eye_direction(frame, poi):
    # Assuming calculate_3d_gaze and polar2cart are defined functions
    theta, pha, delta = calculate_3d_gaze(frame, poi)

    # Convert from polar to Cartesian coordinates for the left eye
    Lx, Ly, Lz = polar2cart(theta[0], pha[0])

    # Convert from polar to Cartesian coordinates for the right eye
    Rx, Ry, Rz = polar2cart(theta[1], pha[1])

    # Calculate the vergence angle from the direction vectors
    angle = vergence_from_direction([Lx, Ly, Lz], [Rx, Ry, Rz])

    return delta, angle, (Lx, Ly, Lz), (Rx, Ry, Rz)


def process_gaze_data_and_vergence_using_eye_arrows(delta, yaw, roll, pupils):
    end_mean_left = [0, 0]
    end_mean_right = [0, 0]
    pi = np.pi

    # Determine the mean based on yaw angle
    if yaw > 30:
        end_mean = delta[0]
    elif yaw < -30:
        end_mean = delta[1]
    else:
        end_mean = np.average(delta, axis=0)
        end_mean_left = delta[0]
        end_mean_right = delta[1]

    # Calculate zeta for the combined average
    if end_mean[0] < 0:
        zeta = np.arctan(end_mean[1] / end_mean[0]) + pi
    else:
        zeta = np.arctan(end_mean[1] / (end_mean[0] + 1e-7))

    # Calculate zeta for the left
    if end_mean_left[0] < 0:
        zeta_left = np.arctan(end_mean_left[1] / end_mean_left[0]) + pi
    else:
        zeta_left = np.arctan(end_mean_left[1] / (end_mean_left[0] + 1e-7))

    # Calculate zeta for the right
    if end_mean_right[0] < 0:
        zeta_right = np.arctan(end_mean_right[1] / end_mean_right[0]) + pi
    else:
        zeta_right = np.arctan(end_mean_right[1] / (end_mean_right[0] + 1e-7))

    # Adjust roll angle
    if roll < 0:
        roll += 180
    else:
        roll -= 180

    # Calculate real angles combining zeta with adjusted roll
    real_angle = zeta + roll * pi / 180
    real_angle_left = zeta_left + roll * pi / 180
    real_angle_right = zeta_right + roll * pi / 180

    # Calculate the offsets for each direction
    R = np.linalg.norm(end_mean)
    offset = (R * np.cos(real_angle), R * np.sin(real_angle))

    R_left = np.linalg.norm(end_mean_left)
    offset_left = (R_left * np.cos(real_angle_left), R_left * np.sin(real_angle_left))

    R_right = np.linalg.norm(end_mean_right)
    offset_right = (
        R_right * np.cos(real_angle_right),
        R_right * np.sin(real_angle_right),
    )

    # Calculate vergence angle from offsets
    angle_arrows = vergence_from_arrows(pupils, offset_left, offset_right)

    return angle_arrows, offset_left, offset_right


def calculate_3d_gaze(frame, poi, scale=256):
    starts, ends, pupils, centers = poi
    # calculated for both left and right eyes:::

    # length between 2 corners of the eyes
    eye_length = norm(starts - ends, axis=1)

    # distance between eye balls and irises in 2D
    ic_distance = norm(pupils - centers, axis=1)

    # distance between eye balls and left corners in 2D
    zc_distance = norm(pupils - starts, axis=1)

    s0 = (starts[:, 1] - ends[:, 1]) * pupils[:, 0]
    s1 = (starts[:, 0] - ends[:, 0]) * pupils[:, 1]
    s2 = starts[:, 0] * ends[:, 1]
    s3 = starts[:, 1] * ends[:, 0]

    delta_y = (s0 - s1 + s2 - s3) / eye_length / 2
    delta_x = np.sqrt(abs(ic_distance**2 - delta_y**2))

    delta = np.array((delta_x * SIN_LEFT_THETA, delta_y * SIN_UP_THETA))
    delta /= eye_length

    theta, pha = np.arcsin(delta)

    # judge variable is in format [False False]
    inv_judge = zc_distance**2 - delta_y**2 < eye_length**2 / 4

    delta[0, inv_judge] *= -1
    theta[inv_judge] *= -1
    delta *= scale

    return theta, pha, delta.T


def polar2cart(theta, pha):
    delta = 1
    return [
        delta * math.sin(theta) * math.cos(pha),
        delta * math.sin(theta) * math.sin(pha),
        delta * math.cos(theta),
    ]


def vergence_from_direction(a, b):
    angle = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return angle


# def vergence_from_arrows(pupils, offset_left, offset_right):
#     a = [(offset_left[0] - pupils[0][0]), (offset_left[1] - pupils[0][1])]
#     b = [(offset_right[0] - pupils[1][0]), (offset_right[1] - pupils[1][1])]

#     angle = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
#     return angle

import numpy as np


def vergence_from_arrows(pupils, offset_left, offset_right):
    # Calculate vectors from pupils to offsets
    a = np.array([offset_left[0] - pupils[0][0], offset_left[1] - pupils[0][1]])
    b = np.array([offset_right[0] - pupils[1][0], offset_right[1] - pupils[1][1]])

    # Check for zero-length vectors to avoid division by zero
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        print("Error: One of the vectors is of zero length.")
        return None

    # Normalize the dot product to avoid out-of-range errors for arccos
    dot_product = np.dot(a, b)
    norms_product = np.linalg.norm(a) * np.linalg.norm(b)
    cos_angle = dot_product / norms_product
    cos_angle = np.clip(
        cos_angle, -1.0, 1.0
    )  # Clipping to handle floating-point precision issues

    # Calculate the angle in radians
    angle = np.arccos(cos_angle)

    return angle


def create_clean_tobii_log(tob):
    df_trial_limits = pd.DataFrame(columns=["trial", "start", "end"])
    df_out = pd.DataFrame(
        columns=[
            "trial_index",
            "timestamp",
            "vergence from tobii in degrees",
            "leftEyePupilSize",
            "rightEyePupilSize",
        ]
    )
    uni_grid = np.arange(0, 2030, 30)

    # filter out NOT A WORD timestamps:
    tob = tob[tob.State != "Mask\n"]

    # filter out validity(comobined of left and right eyes):
    tob = tob[tob.validityScore == "0"]

    # cut out unused columns:
    tob = tob.iloc[:, [0, 6, 7, 23, 26]]

    # make sure tob is in float:
    tob = tob.apply(pd.to_numeric)

    # process every trial:
    for i in range(min(tob.Trial), max(tob.Trial) + 1):
        trial = tob[tob.Trial == i]
        # print("Trail: ", trial)

        # fill in the trial timestamp 'start' and 'end' before normalizing:
        df_trial_limits.loc[len(df_trial_limits)] = [
            i,
            min(trial.TimeStamp),
            max(trial.TimeStamp),
        ]

        # adjust timestamps to 0 start:
        stamps = trial.TimeStamp - min(trial.TimeStamp)
        trial.TimeStamp = stamps

        # start interpolating and making uniform the columns of interest:
        leftEyePupilSize = interp(stamps, list(trial["leftEyePupilSize"]), uni_grid)
        rightEyePupilSize = interp(stamps, list(trial["rightEyePupilSize"]), uni_grid)
        vergence = interp(
            stamps, list(trial["vergence from tobii in degrees"]), uni_grid
        )
        trial_index = [i] * np.ones(len(uni_grid)).astype(int)

        # define a df to append:
        df_append = pd.DataFrame(
            columns=[
                "trial_index",
                "timestamp",
                "vergence from tobii in degrees",
                "leftEyePupilSize",
                "rightEyePupilSize",
            ]
        )
        df_append["trial_index"] = trial_index
        df_append["timestamp"] = uni_grid
        df_append["vergence from tobii in degrees"] = vergence
        df_append["leftEyePupilSize"] = leftEyePupilSize
        df_append["rightEyePupilSize"] = rightEyePupilSize

        # append dataframes:
        df_out = pd.concat([df_out, df_append])
        # print("df_trial_limits: ", df_trial_limits)
    return df_out, df_trial_limits


def create_clean_webcam_log(web, df_trial_limits):
    uni_grid = np.arange(0, 2030, 30)
    df_out = pd.DataFrame(
        columns=[
            "WEB trial_index",
            "WEB timestamp",
            "WEB vergence in degrees",
            "WEB leftEyePupilSize",
            "WEB rightEyePupilSize",
        ]
    )

    web = web.loc[
        :,
        [
            "TimeStamp",
            "Trial",
            "vergence from direction in degrees",
            "pupil left radius",
            "pupil right radius",
        ],
    ]
    for i in range(min(df_trial_limits.trial), max(df_trial_limits.trial) + 1):
        # select start and end timestamp for every trial:
        start_ = df_trial_limits.iloc[i - 1, 1]
        end_ = df_trial_limits.iloc[i - 1, 2]
        # slice out a trial from df:
        web["TimeStamp"] = pd.to_numeric(web["TimeStamp"])
        trial = web[(web["TimeStamp"] >= start_) & (web["TimeStamp"] < end_)]
        # print(trial.TimeStamp)
        # adjust timestamps to 0 start:
        stamps = trial.TimeStamp - min(trial.TimeStamp)
        trial.TimeStamp = stamps
        # start interpolating and making uniform the columns of interest:
        leftEyePupilSize_web = interp(
            stamps, list(trial["pupil left radius"]), uni_grid
        )
        rightEyePupilSize_web = interp(
            stamps, list(trial["pupil right radius"]), uni_grid
        )
        vergence = interp(
            stamps, list(trial["vergence from direction in degrees"]), uni_grid
        )
        trial_index = [i] * np.ones(len(uni_grid)).astype(int)

        # define a df to append:
        df_append = pd.DataFrame(
            columns=[
                "WEB trial_index",
                "WEB timestamp",
                "WEB vergence in degrees",
                "WEB leftEyePupilSize",
                "WEB rightEyePupilSize",
            ]
        )
        df_append["WEB trial_index"] = trial_index
        df_append["WEB timestamp"] = uni_grid
        df_append["WEB vergence in degrees"] = vergence
        df_append["WEB leftEyePupilSize"] = leftEyePupilSize_web
        df_append["WEB rightEyePupilSize"] = rightEyePupilSize_web

        # append dataframes:
        df_out = pd.concat([df_out, df_append])
    return df_out


# def create_clean_webcam_log(web, df_trial_limits):
#     # Define a uniform grid for interpolation
#     uni_grid = np.arange(0, 2030, 30)
#     # Initialize an output DataFrame with defined columns
#     df_out = pd.DataFrame(
#         columns=[
#             "WEB trial_index",
#             "WEB timestamp",
#             "WEB vergence in degrees",
#             "WEB leftEyePupilSize",
#             "WEB rightEyePupilSize",
#         ]
#     )

#     # Select relevant columns from the webcam DataFrame
#     web = web.loc[
#         :,
#         [
#             "TimeStamp",
#             "Trial",
#             "vergence from direction in degrees",
#             "pupil left radius",
#             "pupil right radius",
#         ],
#     ]

#     # Loop over each trial by converting trial indices to integers (corrected to handle floats)
#     for i in range(
#         int(min(df_trial_limits["trial"])), int(max(df_trial_limits["trial"])) + 1
#     ):
#         # Fetch start and end timestamps for each trial using integer indexing
#         start_ = df_trial_limits.iloc[i - 1, 1]
#         end_ = df_trial_limits.iloc[i - 1, 2]

#         # Slice the DataFrame for the current trial and reset index to avoid SettingWithCopyWarning
#         trial = web[(web["TimeStamp"] >= start_) & (web["TimeStamp"] < end_)].copy()
#         # Calculate new timestamps as differences from the minimum timestamp
#         stamps = trial["TimeStamp"] - min(trial["TimeStamp"])
#         trial["TimeStamp"] = stamps

#         # Interpolate data points to the uniform grid defined earlier
#         leftEyePupilSize_web = interp(
#             stamps, list(trial["pupil left radius"]), uni_grid
#         )
#         rightEyePupilSize_web = interp(
#             stamps, list(trial["pupil right radius"]), uni_grid
#         )
#         vergence = interp(
#             stamps, list(trial["vergence from direction in degrees"]), uni_grid
#         )
#         trial_index = [i] * len(uni_grid)

#         # Create a DataFrame for the interpolated data of the current trial
#         df_append = pd.DataFrame(
#             {
#                 "WEB trial_index": trial_index,
#                 "WEB timestamp": uni_grid,
#                 "WEB vergence in degrees": vergence,
#                 "WEB leftEyePupilSize": leftEyePupilSize_web,
#                 "WEB rightEyePupilSize": rightEyePupilSize_web,
#             }
#         )

#         # Append the current trial data to the output DataFrame
#         df_out = pd.concat([df_out, df_append], ignore_index=True)

#     return df_out


# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead

# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   self[name] = value
# Traceback (most recent call last):
#   File "/Users/molinduachintha/Downloads/code for easy execution/main.py", line 39, in <module>
#     main(path2logs)
#   File "/Users/molinduachintha/Downloads/code for easy execution/main.py", line 29, in main
#     final_comparison = subject.compare_vergences(tracker, vergence_csv, game)
#   File "/Users/molinduachintha/Downloads/code for easy execution/BRAINGAZE_class.py", line 168, in compare_vergences
#     webcam_clean = create_clean_webcam_log(web, df_trial_limits)
#   File "/Users/molinduachintha/Downloads/code for easy execution/help_functions.py", line 499, in create_clean_webcam_log
#     trial = web[(web["TimeStamp"] >= start_) & (web["TimeStamp"] < end_)].copy()
#   File "/Users/molinduachintha/opt/anaconda3/envs/mxnet_py38/lib/python3.8/site-packages/pandas/core/ops/common.py", line 69, in new_method
#     return method(self, other)
#   File "/Users/molinduachintha/opt/anaconda3/envs/mxnet_py38/lib/python3.8/site-packages/pandas/core/arraylike.py", line 52, in __ge__
#     return self._cmp_method(other, operator.ge)
#   File "/Users/molinduachintha/opt/anaconda3/envs/mxnet_py38/lib/python3.8/site-packages/pandas/core/series.py", line 5502, in _cmp_method
#     res_values = ops.comparison_op(lvalues, rvalues, op)
#   File "/Users/molinduachintha/opt/anaconda3/envs/mxnet_py38/lib/python3.8/site-packages/pandas/core/ops/array_ops.py", line 284, in comparison_op
#     res_values = comp_method_OBJECT_ARRAY(op, lvalues, rvalues)
#   File "/Users/molinduachintha/opt/anaconda3/envs/mxnet_py38/lib/python3.8/site-packages/pandas/core/ops/array_ops.py", line 73, in comp_method_OBJECT_ARRAY
#     result = libops.scalar_compare(x.ravel(), y, op)
#   File "pandas/_libs/ops.pyx", line 107, in pandas._libs.ops.scalar_compare
# TypeError: '<=' not supported between instances of 'numpy.ndarray' and 'str'

import os


def plot_vergence_over_time(
    csv_file_path, save_path=None, figsize=(10, 6), title="Vergence Over Time"
):
    """
    Plots vergence metrics over time from a given CSV file and optionally saves the figure.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file containing the required columns.
    save_path : str, optional
        Directory or file path to save the resulting plot image.
        If None, the plot is not saved.
    figsize : tuple, optional
        Figure size in inches (width, height), default is (10, 6).
    title : str, optional
        Title of the plot, default is "Vergence Over Time".

    The CSV file should contain the following columns:
    - "timestamp (s)"
    - "vergence from direction in radians"
    - "vergence from direction in degrees"
    - "vergence from gaze arrows in radians"
    - "vergence from gaze arrows in degrees"

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axes objects of the created plot.
    """

    # Load the DataFrame from CSV
    df = pd.read_csv(csv_file_path)

    # Create a new figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each vergence metric if columns exist
    if "timestamp (s)" in df:
        if "vergence from direction in radians" in df:
            ax.plot(
                df["timestamp (s)"],
                df["vergence from direction in radians"],
                label="Direction (rad)",
                color="red",
            )

        if "vergence from direction in degrees" in df:
            ax.plot(
                df["timestamp (s)"],
                df["vergence from direction in degrees"],
                label="Direction (deg)",
                color="blue",
            )

        if "vergence from gaze arrows in radians" in df:
            ax.plot(
                df["timestamp (s)"],
                df["vergence from gaze arrows in radians"],
                label="Arrows (rad)",
                color="green",
            )

        if "vergence from gaze arrows in degrees" in df:
            ax.plot(
                df["timestamp (s)"],
                df["vergence from gaze arrows in degrees"],
                label="Arrows (deg)",
                color="purple",
            )
    else:
        raise ValueError("The DataFrame does not contain 'timestamp (s)' column.")

    # Set labels and title
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vergence")
    ax.set_title(title)

    # Show a legend
    ax.legend()

    # Add a grid
    ax.grid(True)

    # If a save_path is provided, save the figure
    if save_path is not None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    return fig, ax


def plot_vergence_by_unit(df, save_path=None, figsize=(10, 6), title_suffix=""):
    """
    Plots vergence metrics separately for radians and degrees from a given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the required columns.
    save_path : str, optional
        Base directory or file path to save the resulting plot images.
    figsize : tuple, optional
        Figure size in inches (width, height), default is (10, 6).
    title_suffix : str, optional
        Additional suffix for the plot titles.

    Returns
    -------
    None
    """

    # Plotting radians
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(
        df["timestamp (s)"],
        df["vergence from direction in radians"],
        label="Direction (rad)",
        color="red",
    )
    ax1.plot(
        df["timestamp (s)"],
        df["vergence from gaze arrows in radians"],
        label="Arrows (rad)",
        color="green",
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Vergence (Radians)")
    ax1.set_title("Vergence Over Time in Radians" + title_suffix)
    ax1.legend()
    ax1.grid(True)
    if save_path:
        radians_plot_path = os.path.join(save_path, "vergence_radians_plot.png")
        plt.savefig(radians_plot_path)
    plt.show()

    # Plotting degrees
    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.plot(
        df["timestamp (s)"],
        df["vergence from direction in degrees"],
        label="Direction (deg)",
        color="blue",
    )
    ax2.plot(
        df["timestamp (s)"],
        df["vergence from gaze arrows in degrees"],
        label="Arrows (deg)",
        color="purple",
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Vergence (Degrees)")
    ax2.set_title("Vergence Over Time in Degrees" + title_suffix)
    ax2.legend()
    ax2.grid(True)
    if save_path:
        degrees_plot_path = os.path.join(save_path, "vergence_degrees_plot.png")
        plt.savefig(degrees_plot_path)
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot_vergence_by_unit(
    csv_file_path, save_path=None, figsize=(10, 6), title_suffix=""
):
    """
    Plots vergence metrics separately for radians and degrees from a given CSV file.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file containing the required columns.
    save_path : str, optional
        Base directory or file path to save the resulting plot images.
    figsize : tuple, optional
        Figure size in inches (width, height), default is (10, 6).
    title_suffix : str, optional
        Additional suffix for the plot titles.

    Returns
    -------
    None
    """

    # Load data from CSV
    df = pd.read_csv(csv_file_path)

    # Plotting radians
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(
        df["timestamp (s)"],
        df["vergence from direction in radians"],
        label="Direction (rad)",
        color="red",
    )
    ax1.plot(
        df["timestamp (s)"],
        df["vergence from gaze arrows in radians"],
        label="Arrows (rad)",
        color="green",
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Vergence (Radians)")
    ax1.set_title("Vergence Over Time in Radians" + title_suffix)
    ax1.legend()
    ax1.grid(True)
    if save_path:
        radians_plot_path = os.path.join(save_path, "vergence_radians_plot.png")
        os.makedirs(os.path.dirname(radians_plot_path), exist_ok=True)
        plt.savefig(radians_plot_path)
    plt.show()

    # Plotting degrees
    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.plot(
        df["timestamp (s)"],
        df["vergence from direction in degrees"],
        label="Direction (deg)",
        color="blue",
    )
    ax2.plot(
        df["timestamp (s)"],
        df["vergence from gaze arrows in degrees"],
        label="Arrows (deg)",
        color="purple",
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Vergence (Degrees)")
    ax2.set_title("Vergence Over Time in Degrees" + title_suffix)
    ax2.legend()
    ax2.grid(True)
    if save_path:
        degrees_plot_path = os.path.join(save_path, "vergence_degrees_plot.png")
        os.makedirs(os.path.dirname(degrees_plot_path), exist_ok=True)
        plt.savefig(degrees_plot_path)
