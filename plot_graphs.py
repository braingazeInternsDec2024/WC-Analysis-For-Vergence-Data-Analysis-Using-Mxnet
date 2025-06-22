import csv
import json
import os
import glob
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.signal import medfilt, convolve, welch

# def get_nearest_frame(csv_filepath, time_str):
#     """
#     Find the closest frame number to a given timestamp in HH:MM:SS.sss format.

#     Args:
#         csv_filepath (str): Path to the CSV file
#         time_str (str): Target time in "HH:MM:SS.sss" format

#     Returns:
#         int: Frame number with the closest timestamp to the target time
#     """
#     def parse_time(time_str):
#         parts = time_str.split(':')
#         if len(parts) != 3:
#             raise ValueError("Invalid time format. Use HH:MM:SS.sss")

#         hours, minutes, seconds_part = parts
#         if '.' in seconds_part:
#             seconds, milliseconds = seconds_part.split('.')
#             milliseconds = milliseconds.ljust(3, '0')[:3]  # Handle short/long fractions
#         else:
#             seconds = seconds_part
#             milliseconds = '000'

#         return (int(hours) * 3600 +
#                 int(minutes) * 60 +
#                 int(seconds) +
#                 int(milliseconds) / 1000)

#     target_time = parse_time(time_str)
#     closest_frame = None
#     min_difference = float('inf')

#     with open(csv_filepath, 'r') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             current_time = float(row['timestamp (s)'])
#             time_diff = abs(current_time - target_time)

#             if time_diff < min_difference:
#                 min_difference = time_diff
#                 closest_frame = int(row['frame number'])

#     return closest_frame

def extract_time_stamps(csv_filepath, json_filepath=None): 
    df = pd.read_csv(csv_filepath)
    return df["timestamp (s)"].tolist()

def time_to_seconds(time_str):
    """Convert 'HH:MM:SS.sss' to seconds."""
    time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6

def extract_fruit_timings(json_file):
    """Extract fruit timings categorized by fruit type from JSON."""
    if not json_file or not os.path.exists(json_file):
        return {}

    with open(json_file, "r") as f:
        data = json.load(f)

    fruit_timings = {}
    for item in data.get("fruitTimings", []):
        fruit = item["fruit"].lower()
        if fruit not in fruit_timings:
            fruit_timings[fruit] = []
        fruit_timings[fruit].append(time_to_seconds(item["appeared"]))

    return fruit_timings

def find_nearest_frame(time_point, time_stamps):
    """Find the closest frame number for a given timestamp."""
    frame_array = np.array(time_stamps)
    index = (np.abs(frame_array - time_point)).argmin()
    return index

def get_session_name(csv_file):
    """Extract session name from CSV filename, removing .csv extension."""
    base_name = os.path.basename(csv_file)  # Get file name only
    session_name_without_extension = base_name.split(".csv")[0] # Split by ".csv" and take the first part
    session_name = "_".join(session_name_without_extension.split("_")[4:6])
    return session_name

# Smoothing Functions
def moving_average(data, window_size=5):
    """Apply a simple moving average filter."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def exponential_smoothing(data, alpha=0.3):
    """Apply exponential smoothing."""
    smoothed = [data[0]]  # First value remains unchanged
    for i in range(1, len(data)):
        smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)

def gaussian_smoothing(data, sigma=1):
    """Apply Gaussian filter for smoothing."""
    return gaussian_filter1d(data, sigma=sigma)

def savitzky_golay_smoothing(data, window_length=5, polyorder=2):
    """Apply Savitzky-Golay filter for smoothing."""
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)

def median_smoothing(data, size=5):
    """Apply median filter for smoothing."""
    # Note: Median smoothing is typically applied over a window, but for point-wise application, it might not be directly suitable.
    # Returning the median of the entire data series as a placeholder. Consider window-based median filtering if needed.
    return np.median(data)

def moving_median_avg_filter(data):
    median_window_size = 31
    moving_avg_window_size = 31

    moving_median_filtered_signal = medfilt(data, kernel_size=median_window_size)
    moving_median_avg_filtered_signal = moving_average_filter(moving_median_filtered_signal, window_size=moving_avg_window_size)
    return moving_median_avg_filtered_signal

def apply_smoothing(data, technique="moving_average"):
    """Apply the selected smoothing technique."""
    if technique == "moving_average":
        return moving_average(data)
    elif technique == "exponential":
        return exponential_smoothing(data)
    elif technique == "gaussian":
        return gaussian_smoothing(data)
    elif technique == "savgol":
        return savitzky_golay_smoothing(data)
    elif technique == "median":
        return median_smoothing(data) 
    elif technique == "moving_median_avg_filter":
        return moving_median_avg_filter(data)
    else:
        return data  # No smoothing
    
def moving_average_filter(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return convolve(data, window, mode='valid')


def plot_fruit_based_csv(csv_file, json_file, output_base_dir, smoothing_technique=None):
    """Generate fruit-specific plots, both original and smoothed, as separate plots for direction and gaze_arrows."""
    df = pd.read_csv(csv_file)
    df["timestamp (s)"] = pd.to_numeric(df["timestamp (s)"], errors="coerce")
    df.dropna(inplace=True)
    df.sort_values("timestamp (s)", inplace=True)

    fruit_timings = extract_fruit_timings(json_file)
    time_stamps = extract_time_stamps(csv_file)

    session_name = get_session_name(csv_file)
    session_dir = os.path.join(output_base_dir, session_name)

    original_dir = os.path.join(session_dir, "not smoothened")
    smoothed_dir = os.path.join(session_dir, f"smoothened_using_{smoothing_technique}") if smoothing_technique else None

    os.makedirs(original_dir, exist_ok=True)
    if smoothed_dir:
        os.makedirs(smoothed_dir, exist_ok=True)

    for fruit, timings in fruit_timings.items():
        for analysis_type in ["direction", "gaze arrows"]:
            original_fruit_folder = os.path.join(original_dir, analysis_type, fruit)
            smoothed_fruit_folder = os.path.join(smoothed_dir, analysis_type, fruit) if smoothed_dir else None

            os.makedirs(original_fruit_folder, exist_ok=True)
            if smoothed_fruit_folder:
                os.makedirs(smoothed_fruit_folder, exist_ok=True)

            for i, start in enumerate(timings):
                start_idx = find_nearest_frame(start, time_stamps) + 1
                frame_start = max(1, start_idx - 30)
                frame_end = min(len(time_stamps) - 1, start_idx + 60)
                df_window = df.iloc[frame_start:frame_end].copy()

                df_window["frame_number"] = np.arange(len(df_window))

                column_name = f"vergence from {analysis_type} in degrees"

                if column_name not in df.columns:
                    continue  # Skip if data is missing

                # --- Plotting and Saving (Separate Plots) ---
                # **Original Data Plot**
                csv_filename_original = os.path.join(original_fruit_folder, f"{fruit}_event_{i+1}_{analysis_type}.csv") # Corrected CSV filename
                plot_filename_original = os.path.join(original_fruit_folder, f"{fruit}_event_{i+1}_original_{analysis_type}.png") # Corrected plot filename

                df_window[["frame_number", column_name]].to_csv(csv_filename_original, index=False)

                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(df_window["frame_number"], df_window[column_name], color="blue")
                ax.set_title(f"{fruit.capitalize()} Event {i+1} - Vergence from {analysis_type.capitalize()} (Original)") # Updated title
                ax.set_xlabel("Frame Number")
                ax.set_ylabel("Vergence (degrees)")
                ax.grid(True)
                fig.savefig(plot_filename_original)
                plt.close(fig)

                # **Smoothed Data Plot (if applicable)**
                if smoothed_fruit_folder:
                    smoothed_values = apply_smoothing(df_window[column_name].values, technique=smoothing_technique)
                    df_window = df_window.iloc[:len(smoothed_values)]  # Adjust length after smoothing
                    df_window[column_name] = smoothed_values

                    smoothed_csv_filename = os.path.join(smoothed_fruit_folder, f"{fruit}_event_{i+1}_{analysis_type}.csv") # Corrected CSV filename for smoothed
                    smoothed_plot_filename = os.path.join(smoothed_fruit_folder, f"{fruit}_event_{i+1}_smoothed_{analysis_type}.png") # Corrected plot filename for smoothed

                    df_window[["frame_number", column_name]].to_csv(smoothed_csv_filename, index=False)

                    fig, ax = plt.subplots(figsize=(14, 5))
                    ax.plot(df_window["frame_number"], df_window[column_name], color="red")
                    ax.set_title(f"{fruit.capitalize()} Event {i+1} - Vergence from {analysis_type.capitalize()} (Smoothed - {smoothing_technique})") # Updated title for smoothed
                    ax.set_xlabel("Frame Number")
                    ax.set_ylabel("Vergence (degrees)")
                    ax.grid(True)
                    fig.savefig(smoothed_plot_filename)
                    plt.close(fig)

# Function to normalize session name by removing any special characters or spaces
def normalize_session_name(session_name):
    # Remove any special characters (including invisible characters)
    normalized_name = unicodedata.normalize('NFKD', session_name).encode('ascii', 'ignore').decode('ascii')
    # Replace spaces with underscores as well, to ensure consistent folder naming (optional, but good practice)
    normalized_name = normalized_name.replace(" ", "_")
    return normalized_name.strip()

# Function to extract the session name from the CSV filename
def extract_session_name_from_csv(csv_filepath):
    # Assuming the session name starts after 'Vergence_Combined_Calculation_of_' and ends before the timestamp
    filename = os.path.basename(csv_filepath)
    session_name_part = filename.split("Vergence_Combined_Calculation_of_")[1].split(".csv")[0].strip()
    # Normalize the session name part directly after extraction
    normalized_session_name = normalize_session_name(session_name_part)
    return normalized_session_name

# Function to process all CSV files in the directory and call plot_fruit_based_csv
def process_all_csv_in_directory(csv_directory, json_directory, output_directory, smoothing_technique):
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(csv_directory, '**', '*.csv'), recursive=True)

    for csv_filepath in csv_files:
        # Extract and normalize session name from the CSV filepath
        session_name = extract_session_name_from_csv(csv_filepath)

        # **--- DEBUG PRINT STATEMENT ---**
        print(f"DEBUG: Extracted session_name: '{session_name}'")
        # **--- DEBUG PRINT STATEMENT ---**

        # Use the normalized session_name directly to construct session_dir and json_filepath
        session_dir = os.path.join(output_directory, session_name)
        json_filepath = os.path.join(json_directory, session_name, f"StimuliResponseData_{session_name.split('_')[-1]}.json") # Use session_name directly

        # Print session name and JSON path for debugging
        print(f"Extracted session name: '{session_name}'") # Keep this original print for comparison
        print(f"Looking for JSON file at: {json_filepath}")

        # Check if the JSON file exists
        if os.path.exists(json_filepath):
            # Call plot_fruit_based_csv for each CSV with the corresponding JSON file
            plot_fruit_based_csv(csv_filepath, json_filepath, output_directory, smoothing_technique)
        else:
            print(f"Warning: JSON file for session '{session_name}' not found.")


# Example Usage:
csv_directory = "outputs with 100 trials"
json_directory = "data with 100 trials"
output_directory = "output_plots"
smoothing_technique = "moving_median_avg_filter"  # Change to "exponential", "gaussian", "savgol", "median", or None for no smoothing

process_all_csv_in_directory(csv_directory, json_directory, output_directory, smoothing_technique)