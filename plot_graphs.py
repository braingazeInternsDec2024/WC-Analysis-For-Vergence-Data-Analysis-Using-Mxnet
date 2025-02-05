import csv
import json

def get_nearest_frame(csv_filepath, time_str):
    """
    Find the closest frame number to a given timestamp in HH:MM:SS.sss format.
    
    Args:
        csv_filepath (str): Path to the CSV file
        time_str (str): Target time in "HH:MM:SS.sss" format
    
    Returns:
        int: Frame number with the closest timestamp to the target time
    """
    def parse_time(time_str):
        parts = time_str.split(':')
        if len(parts) != 3:
            raise ValueError("Invalid time format. Use HH:MM:SS.sss")
        
        hours, minutes, seconds_part = parts
        if '.' in seconds_part:
            seconds, milliseconds = seconds_part.split('.')
            milliseconds = milliseconds.ljust(3, '0')[:3]  # Handle short/long fractions
        else:
            seconds = seconds_part
            milliseconds = '000'
            
        return (int(hours) * 3600 + 
                int(minutes) * 60 + 
                int(seconds) + 
                int(milliseconds) / 1000)

    target_time = parse_time(time_str)
    closest_frame = None
    min_difference = float('inf')

    with open(csv_filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            current_time = float(row['timestamp (s)'])
            time_diff = abs(current_time - target_time)
            
            if time_diff < min_difference:
                min_difference = time_diff
                closest_frame = int(row['frame number'])
                
    return closest_frame

def extract_frame_numbers(csv_filepath, json_filepath):
    """
    Extract the nearest frame numbers for each 'appeared' time in the fruit timings from the JSON data.
    
    Args:
        csv_filepath (str): Path to the CSV file containing timestamp and frame number data.
        json_filepath (str): Path to the JSON file containing 'fruitTimings' with 'appeared' times.
    
    Returns:
        list: Array of frame numbers corresponding to the closest frames for each 'appeared' time.
    """
    # Load JSON data from the file
    with open(json_filepath, 'r') as file:
        json_data = json.load(file)
    
    frame_numbers = []
    for timing in json_data.get("fruitTimings", []):
        appeared_time = timing.get("appeared")
        if appeared_time:
            frame = get_nearest_frame(csv_filepath, appeared_time)
            frame_numbers.append(frame)
    return frame_numbers


import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

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

def extract_frame_numbers(csv_file):
    """Extract frame numbers from CSV."""
    df = pd.read_csv(csv_file)
    return df["timestamp (s)"].tolist()

def find_nearest_frame(time_point, frame_numbers):
    """Find the closest frame number for a given timestamp."""
    frame_array = np.array(frame_numbers)
    index = (np.abs(frame_array - time_point)).argmin()
    return index

def get_session_name(csv_file):
    """Extract session name from CSV filename."""
    base_name = os.path.basename(csv_file)  # Get file name only
    session_name = "_".join(base_name.split("_")[4:6])  # Extract "Molindu_20250205080611"
    return session_name

# def plot_fruit_based_csv(csv_file, json_file, output_base_dir):
#     """Generate fruit-specific plots and store corresponding CSVs inside 'direction' and 'gaze_arrows' folders."""
#     df = pd.read_csv(csv_file)
#     df["timestamp (s)"] = pd.to_numeric(df["timestamp (s)"], errors="coerce")
#     df.dropna(inplace=True)
#     df.sort_values("timestamp (s)", inplace=True)
    
#     fruit_timings = extract_fruit_timings(json_file)
#     frame_numbers = extract_frame_numbers(csv_file)
    
#     # Get session folder name and create directories
#     session_name = get_session_name(csv_file)
#     session_dir = os.path.join(output_base_dir, session_name)
#     direction_dir = os.path.join(session_dir, "direction")
#     gaze_arrows_dir = os.path.join(session_dir, "gaze_arrows")

#     os.makedirs(direction_dir, exist_ok=True)
#     os.makedirs(gaze_arrows_dir, exist_ok=True)

#     for fruit, timings in fruit_timings.items():
#         # Create fruit subdirectories for both analyses
#         fruit_direction_folder = os.path.join(direction_dir, fruit)
#         fruit_gaze_arrows_folder = os.path.join(gaze_arrows_dir, fruit)

#         os.makedirs(fruit_direction_folder, exist_ok=True)
#         os.makedirs(fruit_gaze_arrows_folder, exist_ok=True)
        
#         for i, start in enumerate(timings):
#             start_idx = find_nearest_frame(start, frame_numbers)
#             frame_start = max(0, start_idx - 10)
#             frame_end = min(len(frame_numbers) - 1, start_idx + 75)
#             df_window = df.iloc[frame_start:frame_end].copy()  # Create a copy to modify safely
            
#             # Create frame index starting from 0
#             df_window["frame_number"] = np.arange(len(df_window))

#             # **Save CSV & Plot for Vergence from Direction**
#             csv_filename_direction = os.path.join(fruit_direction_folder, f"{fruit}_event_{i+1}.csv")
#             plot_filename_direction = os.path.join(fruit_direction_folder, f"{fruit}_event_{i+1}.png")

#             df_window[["frame_number", "vergence from direction in degrees"]].to_csv(csv_filename_direction, index=False)

#             fig, ax = plt.subplots(figsize=(14, 5))
#             ax.plot(df_window["frame_number"], df_window["vergence from direction in degrees"], color="blue")
#             ax.set_title(f"{fruit.capitalize()} Event {i+1} - Vergence from Direction")
#             ax.set_xlabel("Frame Number")
#             ax.set_ylabel("Vergence (degrees)")
#             ax.grid(True)
#             fig.savefig(plot_filename_direction)
#             plt.close(fig)

#             # **Save CSV & Plot for Vergence from Gaze Arrows (if column exists)**
#             if "vergence from gaze arrows in degrees" in df.columns:
#                 csv_filename_gaze_arrows = os.path.join(fruit_gaze_arrows_folder, f"{fruit}_event_{i+1}.csv")
#                 plot_filename_gaze_arrows = os.path.join(fruit_gaze_arrows_folder, f"{fruit}_event_{i+1}.png")

#                 df_window[["frame_number", "vergence from gaze arrows in degrees"]].to_csv(csv_filename_gaze_arrows, index=False)

#                 fig, ax = plt.subplots(figsize=(14, 5))
#                 ax.plot(df_window["frame_number"], df_window["vergence from gaze arrows in degrees"], color="red")
#                 ax.set_title(f"{fruit.capitalize()} Event {i+1} - Vergence from Gaze Arrows")
#                 ax.set_xlabel("Frame Number")
#                 ax.set_ylabel("Vergence (degrees)")
#                 ax.grid(True)
#                 fig.savefig(plot_filename_gaze_arrows)
#                 plt.close(fig)

# # Example Usage
# csv_filepath = "outputs with 100 trials/Vergence_Combined_Calculation_of_Molindu_20250205080611.csv"
# json_filepath = "data with 100 trials/Molindu_20250205080611/StimuliResponseData_20250205080611.json"
# output_directory = "output_plots"

# plot_fruit_based_csv(csv_filepath, json_filepath, output_directory)


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

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
    return np.median(data)

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
    else:
        return data  # No smoothing

def plot_fruit_based_csv(csv_file, json_file, output_base_dir, smoothing_technique=None):
    """Generate fruit-specific plots, both original and smoothed."""
    df = pd.read_csv(csv_file)
    df["timestamp (s)"] = pd.to_numeric(df["timestamp (s)"], errors="coerce")
    df.dropna(inplace=True)
    df.sort_values("timestamp (s)", inplace=True)

    fruit_timings = extract_fruit_timings(json_file)
    frame_numbers = extract_frame_numbers(csv_file)

    session_name = get_session_name(csv_file)
    session_dir = os.path.join(output_base_dir, session_name)
    
    # Create "not smoothened" (original) and "smoothened_using_<technique>" directories
    original_dir = os.path.join(session_dir, "not smoothened")
    smoothed_dir = os.path.join(session_dir, f"smoothened_using_{smoothing_technique}") if smoothing_technique else None

    os.makedirs(original_dir, exist_ok=True)
    if smoothed_dir:
        os.makedirs(smoothed_dir, exist_ok=True)

    for fruit, timings in fruit_timings.items():
        for analysis_type in ["direction", "gaze_arrows"]:
            original_fruit_folder = os.path.join(original_dir, analysis_type, fruit)
            smoothed_fruit_folder = os.path.join(smoothed_dir, analysis_type, fruit) if smoothed_dir else None

            os.makedirs(original_fruit_folder, exist_ok=True)
            if smoothed_fruit_folder:
                os.makedirs(smoothed_fruit_folder, exist_ok=True)
        
            for i, start in enumerate(timings):
                start_idx = find_nearest_frame(start, frame_numbers)
                frame_start = max(0, start_idx - 10)
                frame_end = min(len(frame_numbers) - 1, start_idx + 75)
                df_window = df.iloc[frame_start:frame_end].copy()
                
                df_window["frame_number"] = np.arange(len(df_window))
                
                # Define columns for vergence from direction and gaze arrows
                column_name_direction = "vergence from direction in degrees"
                column_name_gaze_arrows = "vergence from gaze arrows in degrees"
                
                if column_name_direction not in df.columns or column_name_gaze_arrows not in df.columns:
                    continue  # Skip if data is missing

                # Save original data for direction and gaze arrows
                csv_filename_direction = os.path.join(original_fruit_folder, f"{fruit}_event_{i+1}_direction.csv")
                csv_filename_gaze_arrows = os.path.join(original_fruit_folder, f"{fruit}_event_{i+1}_gaze_arrows.csv")
                
                df_window[["frame_number", column_name_direction]].to_csv(csv_filename_direction, index=False)
                df_window[["frame_number", column_name_gaze_arrows]].to_csv(csv_filename_gaze_arrows, index=False)

                # Plot original data for direction and gaze arrows
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(df_window["frame_number"], df_window[column_name_direction], color="blue", label="Vergence from Direction")
                ax.plot(df_window["frame_number"], df_window[column_name_gaze_arrows], color="green", label="Vergence from Gaze Arrows")
                ax.set_title(f"{fruit.capitalize()} Event {i+1} - Original Data")
                ax.set_xlabel("Frame Number")
                ax.set_ylabel("Vergence (degrees)")
                ax.grid(True)
                ax.legend()
                fig.savefig(os.path.join(original_fruit_folder, f"{fruit}_event_{i+1}_original.png"))
                plt.close(fig)

                # Apply smoothing if selected
                if smoothed_fruit_folder:
                    smoothed_values_direction = apply_smoothing(df_window[column_name_direction].values, technique=smoothing_technique)
                    smoothed_values_gaze_arrows = apply_smoothing(df_window[column_name_gaze_arrows].values, technique=smoothing_technique)
                    
                    df_window = df_window.iloc[:len(smoothed_values_direction)]  # Adjust length after smoothing
                    df_window[column_name_direction] = smoothed_values_direction
                    df_window[column_name_gaze_arrows] = smoothed_values_gaze_arrows

                    smoothed_csv_filename_direction = os.path.join(smoothed_fruit_folder, f"{fruit}_event_{i+1}_direction.csv")
                    smoothed_csv_filename_gaze_arrows = os.path.join(smoothed_fruit_folder, f"{fruit}_event_{i+1}_gaze_arrows.csv")
                    
                    df_window[["frame_number", column_name_direction]].to_csv(smoothed_csv_filename_direction, index=False)
                    df_window[["frame_number", column_name_gaze_arrows]].to_csv(smoothed_csv_filename_gaze_arrows, index=False)

                    # Plot smoothed data for direction and gaze arrows
                    fig, ax = plt.subplots(figsize=(14, 5))
                    ax.plot(df_window["frame_number"], df_window[column_name_direction], color="blue", label="Vergence from Direction (Smoothed)")
                    ax.plot(df_window["frame_number"], df_window[column_name_gaze_arrows], color="green", label="Vergence from Gaze Arrows (Smoothed)")
                    ax.set_title(f"{fruit.capitalize()} Event {i+1} - Smoothed Data ({smoothing_technique})")
                    ax.set_xlabel("Frame Number")
                    ax.set_ylabel("Vergence (degrees)")
                    ax.grid(True)
                    ax.legend()
                    fig.savefig(os.path.join(smoothed_fruit_folder, f"{fruit}_event_{i+1}_smoothed.png"))
                    plt.close(fig)


# csv_filepath = "outputs with 100 trials/Vergence_Combined_Calculation_of_Molindu A​_20250204163331.csv"
# json_filepath = "data with 100 trials/Molindu A​_20250204163331/StimuliResponseData_20250204163331.json"
# output_directory = "output_plots"
# smoothing_technique = "moving_average"  # Change to "exponential", "gaussian", "savgol", "median", or None for no smoothing

# plot_fruit_based_csv(csv_filepath, json_filepath, output_directory, smoothing_technique)

import os
import glob
import unicodedata

# Function to normalize session name by removing any special characters or spaces
def normalize_session_name(session_name):
    # Remove any special characters (including invisible characters)
    normalized_name = unicodedata.normalize('NFKD', session_name).encode('ascii', 'ignore').decode('ascii')
    return normalized_name.strip()

# Function to extract the session name from the CSV filename
def extract_session_name_from_csv(csv_filepath):
    # Assuming the session name starts after 'Vergence_Combined_Calculation_of_' and ends before the timestamp
    filename = os.path.basename(csv_filepath)
    session_name = filename.split("Vergence_Combined_Calculation_of_")[1].split(".csv")[0].strip()  # Added strip()
    # Normalize the session name to remove special characters or invisible spaces
    return normalize_session_name(session_name)

# Function to process all CSV files in the directory and call plot_fruit_based_csv
def process_all_csv_in_directory(csv_directory, json_directory, output_directory, smoothing_technique):
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(csv_directory, '**', '*.csv'), recursive=True)
    
    for csv_filepath in csv_files:
        # Extract and normalize session name from the CSV filepath
        session_name = extract_session_name_from_csv(csv_filepath)
        
        # Construct the corresponding folder name by removing the .csv extension
        folder_name = session_name.split('_')[0] + "_" + session_name.split('_')[1]  # Format as 'SessionName_Timestamp'
        
        # Remove any '.csv' extension if present
        folder_name = folder_name.rstrip('.csv')
        
        json_filepath = os.path.join(json_directory, folder_name, f"StimuliResponseData_{session_name.split('_')[1]}.json")
        
        # Print session name and JSON path for debugging
        print(f"Extracted session name: '{session_name}'")
        print(f"Looking for JSON file at: {json_filepath}")
        
        # Check if the JSON file exists
        if os.path.exists(json_filepath):
            # Call plot_fruit_based_csv for each CSV with the corresponding JSON file
            plot_fruit_based_csv(csv_filepath, json_filepath, output_directory, smoothing_technique)
        else:
            print(f"Warning: JSON file for session '{session_name}' not found.")


csv_directory = "outputs with 100 trials"
json_directory = "data with 100 trials"
output_directory = "output_plots"
smoothing_technique = "moving_average"  # Change as needed

process_all_csv_in_directory(csv_directory, json_directory, output_directory, smoothing_technique)




