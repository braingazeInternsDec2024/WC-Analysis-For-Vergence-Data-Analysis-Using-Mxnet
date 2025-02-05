# import os
# import glob
# import pandas as pd
# import matplotlib.pyplot as plt


# def plot_all_csv_in_dir(input_dir):
#     """
#     Find all CSVs named 'Vergence_Combined_Calculation_of_*.csv' in 'input_dir'
#     and generate four plots for each CSV:
#       1) Direction (no smoothing)
#       2) Direction (with smoothing)
#       3) Gaze arrows (no smoothing)
#       4) Gaze arrows (with smoothing)

#     The plots are saved under:
#       output_plots/
#           direction/
#               no_smoothing/
#               with_smoothing/
#           gaze_arrows/
#               no_smoothing/
#               with_smoothing/
#     """
#     # Main output directory
#     output_dir = "output_plots"
#     os.makedirs(output_dir, exist_ok=True)

#     # Separate subfolders for direction: no smoothing & with smoothing
#     dir_no_smooth = os.path.join(output_dir, "direction", "no_smoothing")
#     dir_smooth = os.path.join(output_dir, "direction", "with_smoothing")
#     os.makedirs(dir_no_smooth, exist_ok=True)
#     os.makedirs(dir_smooth, exist_ok=True)

#     # Separate subfolders for gaze arrows: no smoothing & with smoothing
#     gaze_no_smooth = os.path.join(output_dir, "gaze_arrows", "no_smoothing")
#     gaze_smooth = os.path.join(output_dir, "gaze_arrows", "with_smoothing")
#     os.makedirs(gaze_no_smooth, exist_ok=True)
#     os.makedirs(gaze_smooth, exist_ok=True)

#     # Find matching CSV files
#     pattern = os.path.join(input_dir, "Vergence_Combined_Calculation_of_*.csv")
#     csv_files = glob.glob(pattern)

#     if not csv_files:
#         print(f"No matching CSV files found in {input_dir}.")
#         return

#     for csv_file in csv_files:
#         # Extract the "video name" from the filename
#         filename = os.path.basename(csv_file)
#         video_name = os.path.splitext(filename)[0].replace(
#             "Vergence_Combined_Calculation_of_", ""
#         )
#         print(f"Processing CSV: {filename}  (video: {video_name})")

#         # Generate plots for this CSV
#         plot_single_csv(
#             csv_file,
#             video_name,
#             dir_no_smooth,  # direction (no smoothing)
#             dir_smooth,  # direction (with smoothing)
#             gaze_no_smooth,  # gaze arrows (no smoothing)
#             gaze_smooth,  # gaze arrows (with smoothing)
#         )


# def plot_single_csv(
#     csv_file, video_name, dir_no_smooth, dir_smooth, gaze_no_smooth, gaze_smooth
# ):
#     """
#     Reads a single CSV file with columns:
#       - 'timestamp (s)'
#       - 'vergence from direction in degrees'
#       - 'vergence from gaze arrows in degrees'
#     Produces 4 plots and saves them into their respective subfolders:
#       1) Direction (No Smoothing)      -> dir_no_smooth
#       2) Direction (With Smoothing)    -> dir_smooth
#       3) Gaze Arrows (No Smoothing)    -> gaze_no_smooth
#       4) Gaze Arrows (With Smoothing)  -> gaze_smooth
#     """
#     # Read the CSV
#     df = pd.read_csv(csv_file)

#     # Convert columns to numeric if needed
#     df["timestamp (s)"] = pd.to_numeric(df["timestamp (s)"], errors="coerce")
#     df["vergence from direction in degrees"] = pd.to_numeric(
#         df["vergence from direction in degrees"], errors="coerce"
#     )
#     df["vergence from gaze arrows in degrees"] = pd.to_numeric(
#         df["vergence from gaze arrows in degrees"], errors="coerce"
#     )

#     # Drop rows missing any required data
#     df.dropna(
#         subset=[
#             "timestamp (s)",
#             "vergence from direction in degrees",
#             "vergence from gaze arrows in degrees",
#         ],
#         inplace=True,
#     )

#     # Sort by timestamp (optional, but recommended)
#     df.sort_values("timestamp (s)", inplace=True)

#     # 1) DIRECTION (NO SMOOTHING)
#     fig, ax = plt.subplots(figsize=(14, 5))
#     ax.plot(
#         df["timestamp (s)"],
#         df["vergence from direction in degrees"],
#         label="Direction - No Smoothing",
#         color="blue",
#     )
#     ax.set_title(f"Direction_{video_name} (No Smoothing)")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Vergence (degrees)")
#     ax.legend()
#     ax.grid(True)

#     direction_no_smooth_path = os.path.join(
#         dir_no_smooth, f"Direction_{video_name}_no_smoothing.png"
#     )
#     fig.savefig(direction_no_smooth_path)
#     plt.close(fig)

#     # 2) DIRECTION (WITH SMOOTHING)
#     df["direction_smoothed"] = (
#         df["vergence from direction in degrees"]
#         .rolling(window=10, min_periods=1)
#         .mean()
#     )

#     fig, ax = plt.subplots(figsize=(14, 5))
#     ax.plot(
#         df["timestamp (s)"],
#         df["direction_smoothed"],
#         label="Direction - With Smoothing",
#         color="blue",
#         linewidth=2,
#     )
#     ax.set_title(f"Direction_{video_name} (With Smoothing)")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Vergence (degrees)")
#     ax.legend()
#     ax.grid(True)

#     direction_smooth_path = os.path.join(
#         dir_smooth, f"Direction_{video_name}_with_smoothing.png"
#     )
#     fig.savefig(direction_smooth_path)
#     plt.close(fig)

#     # 3) GAZE ARROWS (NO SMOOTHING)
#     fig, ax = plt.subplots(figsize=(14, 5))
#     ax.plot(
#         df["timestamp (s)"],
#         df["vergence from gaze arrows in degrees"],
#         label="Gaze Arrows - No Smoothing",
#         color="red",
#     )
#     ax.set_title(f"Gaze_Arrows_{video_name} (No Smoothing)")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Vergence (degrees)")
#     ax.legend()
#     ax.grid(True)

#     gaze_no_smooth_path = os.path.join(
#         gaze_no_smooth, f"Gaze_Arrows_{video_name}_no_smoothing.png"
#     )
#     fig.savefig(gaze_no_smooth_path)
#     plt.close(fig)

#     # 4) GAZE ARROWS (WITH SMOOTHING)
#     df["gaze_smoothed"] = (
#         df["vergence from gaze arrows in degrees"]
#         .rolling(window=10, min_periods=1)
#         .mean()
#     )

#     fig, ax = plt.subplots(figsize=(14, 5))
#     ax.plot(
#         df["timestamp (s)"],
#         df["gaze_smoothed"],
#         label="Gaze Arrows - With Smoothing",
#         color="red",
#         linewidth=2,
#     )
#     ax.set_title(f"Gaze_Arrows_{video_name} (With Smoothing)")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Vergence (degrees)")
#     ax.legend()
#     ax.grid(True)

#     gaze_smooth_path = os.path.join(
#         gaze_smooth, f"Gaze_Arrows_{video_name}_with_smoothing.png"
#     )
#     fig.savefig(gaze_smooth_path)
#     plt.close(fig)

#     print(
#         f"Saved 4 plots for '{video_name}' into:\n"
#         f"  - Direction: {dir_no_smooth} and {dir_smooth}\n"
#         f"  - Gaze:      {gaze_no_smooth} and {gaze_smooth}"
#     )


# # -----------------------
# # Example Usage
# # -----------------------
# if __name__ == "__main__":
#     input_dir = "output"
#     plot_all_csv_in_dir(input_dir)


# import os
# import glob
# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime


# def time_to_seconds(time_str):
#     """Convert 'HH:MM:SS.sss' to seconds."""
#     time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
#     return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6


# def extract_fruit_timings(json_file):
#     """Extract fruit appearance and disappearance timings from JSON."""
#     if not json_file or not os.path.exists(json_file):
#         return []  # No JSON file, return empty list

#     with open(json_file, "r") as f:
#         data = json.load(f)

#     fruit_timings = [
#         (time_to_seconds(item["appeared"]), time_to_seconds(item["disappeared"]))
#         for item in data.get("fruitTimings", [])
#     ]

#     return fruit_timings


# def find_json_in_folder(csv_file_path):
#     """Finds the corresponding JSON file inside 'data/<folder_name>/StimuliResponseData_*.json'."""
#     filename = os.path.basename(csv_file_path)
#     folder_name = filename.replace("Vergence_Combined_Calculation_of_", "").replace(".csv", "")

#     json_folder = os.path.join("data", folder_name)  # JSON is stored in 'data/<folder_name>/'

#     if not os.path.exists(json_folder):
#         return None  # Folder does not exist

#     pattern = os.path.join(json_folder, "StimuliResponseData_*.json")
#     matching_jsons = glob.glob(pattern)

#     return matching_jsons[0] if matching_jsons else None  # Return the first match or None


# def plot_all_csv_in_dir(input_dir):
#     """Process all CSVs and generate plots."""
#     output_dir = "output_plots"
#     os.makedirs(output_dir, exist_ok=True)

#     dir_no_smooth = os.path.join(output_dir, "direction", "no_smoothing")
#     dir_smooth = os.path.join(output_dir, "direction", "with_smoothing")
#     gaze_no_smooth = os.path.join(output_dir, "gaze_arrows", "no_smoothing")
#     gaze_smooth = os.path.join(output_dir, "gaze_arrows", "with_smoothing")

#     for d in [dir_no_smooth, dir_smooth, gaze_no_smooth, gaze_smooth]:
#         os.makedirs(d, exist_ok=True)

#     pattern = os.path.join(input_dir, "Vergence_Combined_Calculation_of_*.csv")
#     csv_files = glob.glob(pattern)

#     if not csv_files:
#         print(f"No matching CSV files found in {input_dir}.")
#         return

#     for csv_file in csv_files:
#         filename = os.path.basename(csv_file)
#         folder_name = filename.replace("Vergence_Combined_Calculation_of_", "").replace(".csv", "")

#         json_file = find_json_in_folder(csv_file)
#         fruit_timings = extract_fruit_timings(json_file)

#         print(f"Processing CSV: {filename} (Folder: {folder_name}) with {len(fruit_timings)} shaded regions.")

#         plot_single_csv(csv_file, filename, dir_no_smooth, dir_smooth, gaze_no_smooth, gaze_smooth, fruit_timings)


# def add_shaded_regions(ax, fruit_timings):
#     """Add shaded regions to the plot based on fruit timings."""
#     for timing in fruit_timings:
#         start_time, end_time = timing
#         print(f"Shading from {start_time} to {end_time}")  # Debugging
#         ax.axvspan(start_time, end_time, color="gray", alpha=0.3)


# def plot_single_csv(csv_file, filename, dir_no_smooth, dir_smooth, gaze_no_smooth, gaze_smooth, fruit_timings):
#     """Reads a CSV file and generates 4 plots with shaded fruit appearance regions."""
#     df = pd.read_csv(csv_file)

#     df["timestamp (s)"] = pd.to_numeric(df["timestamp (s)"], errors="coerce")
#     df["vergence from direction in degrees"] = pd.to_numeric(df["vergence from direction in degrees"], errors="coerce")
#     df["vergence from gaze arrows in degrees"] = pd.to_numeric(df["vergence from gaze arrows in degrees"], errors="coerce")

#     df.dropna(inplace=True)
#     df.sort_values("timestamp (s)", inplace=True)

#     # 1) DIRECTION (NO SMOOTHING)
#     fig, ax = plt.subplots(figsize=(14, 5))
#     ax.plot(df["timestamp (s)"], df["vergence from direction in degrees"], label="Direction - No Smoothing", color="blue")
#     add_shaded_regions(ax, fruit_timings)
#     ax.set_title(f"Direction_{filename} (No Smoothing)")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Vergence (degrees)")
#     ax.legend()
#     ax.grid(True)
#     fig.savefig(os.path.join(dir_no_smooth, f"Direction_{filename}_no_smoothing.png"))
#     plt.close(fig)

#     # 2) DIRECTION (WITH SMOOTHING)
#     df["direction_smoothed"] = df["vergence from direction in degrees"].rolling(window=10, min_periods=1).mean()
#     fig, ax = plt.subplots(figsize=(14, 5))
#     ax.plot(df["timestamp (s)"], df["direction_smoothed"], label="Direction - With Smoothing", color="blue", linewidth=2)
#     add_shaded_regions(ax, fruit_timings)
#     ax.set_title(f"Direction_{filename} (With Smoothing)")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Vergence (degrees)")
#     ax.legend()
#     ax.grid(True)
#     fig.savefig(os.path.join(dir_smooth, f"Direction_{filename}_with_smoothing.png"))
#     plt.close(fig)

#     # 3) GAZE ARROWS (NO SMOOTHING)
#     fig, ax = plt.subplots(figsize=(14, 5))
#     ax.plot(df["timestamp (s)"], df["vergence from gaze arrows in degrees"], label="Gaze Arrows - No Smoothing", color="red")
#     add_shaded_regions(ax, fruit_timings)
#     ax.set_title(f"Gaze_{filename} (No Smoothing)")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Vergence (degrees)")
#     ax.legend()
#     ax.grid(True)
#     fig.savefig(os.path.join(gaze_no_smooth, f"Gaze_{filename}_no_smoothing.png"))
#     plt.close(fig)

#     # 4) GAZE ARROWS (WITH SMOOTHING)
#     df["gaze_smoothed"] = df["vergence from gaze arrows in degrees"].rolling(window=10, min_periods=1).mean()
#     fig, ax = plt.subplots(figsize=(14, 5))
#     ax.plot(df["timestamp (s)"], df["gaze_smoothed"], label="Gaze Arrows - With Smoothing", color="red", linewidth=2)
#     add_shaded_regions(ax, fruit_timings)
#     ax.set_title(f"Gaze_{filename} (With Smoothing)")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Vergence (degrees)")
#     ax.legend()
#     ax.grid(True)
#     fig.savefig(os.path.join(gaze_smooth, f"Gaze_{filename}_with_smoothing.png"))
#     plt.close(fig)


import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def time_to_seconds(time_str):
    """Convert 'HH:MM:SS.sss' to seconds."""
    time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6


def extract_fruit_timings_and_button_presses(json_file):
    """Extract fruit timings and button press timestamps from JSON."""
    if not json_file or not os.path.exists(json_file):
        return [], []  # No JSON file, return empty lists

    with open(json_file, "r") as f:
        data = json.load(f)

    fruit_timings = [
        (time_to_seconds(item["appeared"]), time_to_seconds(item["disappeared"]))
        for item in data.get("fruitTimings", [])
    ]

    button_presses = [time_to_seconds(time) for time in data.get("buttonPressedTimes", [])]

    return fruit_timings, button_presses


def find_json_in_folder(csv_file_path):
    """Finds the corresponding JSON file inside 'data/<folder_name>/StimuliResponseData_*.json'."""
    filename = os.path.basename(csv_file_path)
    folder_name = filename.replace("Vergence_Combined_Calculation_of_", "").replace(".csv", "")

    json_folder = os.path.join("data", folder_name)  # JSON is stored in 'data/<folder_name>/'

    if not os.path.exists(json_folder):
        return None  # Folder does not exist

    pattern = os.path.join(json_folder, "StimuliResponseData_*.json")
    matching_jsons = glob.glob(pattern)

    return matching_jsons[0] if matching_jsons else None  # Return the first match or None


def plot_all_csv_in_dir(input_dir):
    """Process all CSVs and generate plots."""
    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)

    dir_no_smooth = os.path.join(output_dir, "direction", "no_smoothing")
    dir_smooth = os.path.join(output_dir, "direction", "with_smoothing")
    gaze_no_smooth = os.path.join(output_dir, "gaze_arrows", "no_smoothing")
    gaze_smooth = os.path.join(output_dir, "gaze_arrows", "with_smoothing")

    for d in [dir_no_smooth, dir_smooth, gaze_no_smooth, gaze_smooth]:
        os.makedirs(d, exist_ok=True)

    pattern = os.path.join(input_dir, "Vergence_Combined_Calculation_of_*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No matching CSV files found in {input_dir}.")
        return

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        folder_name = filename.replace("Vergence_Combined_Calculation_of_", "").replace(".csv", "")

        json_file = find_json_in_folder(csv_file)
        fruit_timings, button_presses = extract_fruit_timings_and_button_presses(json_file)

        print(f"Processing CSV: {filename} (Folder: {folder_name}) with {len(fruit_timings)} shaded regions and {len(button_presses)} button press markers.")

        plot_single_csv(csv_file, filename, dir_no_smooth, dir_smooth, gaze_no_smooth, gaze_smooth, fruit_timings, button_presses)


def add_shaded_regions(ax, fruit_timings):
    """Add shaded regions to the plot based on fruit timings."""
    for start_time, end_time in fruit_timings:
        ax.axvspan(start_time, end_time, color="gray", alpha=0.3)


def add_button_press_lines(ax, button_presses):
    """Add vertical lines for button press events."""
    for time in button_presses:
        ax.axvline(time, color="green", linestyle="--", linewidth=1.5)


def plot_single_csv(csv_file, filename, dir_no_smooth, dir_smooth, gaze_no_smooth, gaze_smooth, fruit_timings, button_presses):
    """Reads a CSV file and generates 4 plots with shaded fruit appearance regions and button press markers."""
    df = pd.read_csv(csv_file)

    df["timestamp (s)"] = pd.to_numeric(df["timestamp (s)"], errors="coerce")
    df["vergence from direction in degrees"] = pd.to_numeric(df["vergence from direction in degrees"], errors="coerce")
    df["vergence from gaze arrows in degrees"] = pd.to_numeric(df["vergence from gaze arrows in degrees"], errors="coerce")

    df.dropna(inplace=True)
    df.sort_values("timestamp (s)", inplace=True)

    # 1) DIRECTION (NO SMOOTHING)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp (s)"], df["vergence from direction in degrees"], color="blue")
    add_shaded_regions(ax, fruit_timings)
    add_button_press_lines(ax, button_presses)
    ax.set_title(f"Direction_{filename} (No Smoothing)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vergence (degrees)")
    ax.grid(True)
    fig.savefig(os.path.join(dir_no_smooth, f"Direction_{filename}_no_smoothing.png"))
    plt.close(fig)

    # 2) DIRECTION (WITH SMOOTHING)
    df["direction_smoothed"] = df["vergence from direction in degrees"].rolling(window=10, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp (s)"], df["direction_smoothed"], color="blue", linewidth=2)
    add_shaded_regions(ax, fruit_timings)
    add_button_press_lines(ax, button_presses)
    ax.set_title(f"Direction_{filename} (With Smoothing)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vergence (degrees)")
    ax.grid(True)
    fig.savefig(os.path.join(dir_smooth, f"Direction_{filename}_with_smoothing.png"))
    plt.close(fig)

    # 3) GAZE ARROWS (NO SMOOTHING)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp (s)"], df["vergence from gaze arrows in degrees"], color="red")
    add_shaded_regions(ax, fruit_timings)
    add_button_press_lines(ax, button_presses)
    ax.set_title(f"Gaze_{filename} (No Smoothing)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vergence (degrees)")
    ax.grid(True)
    fig.savefig(os.path.join(gaze_no_smooth, f"Gaze_{filename}_no_smoothing.png"))
    plt.close(fig)

    # 4) GAZE ARROWS (WITH SMOOTHING)
    df["gaze_smoothed"] = df["vergence from gaze arrows in degrees"].rolling(window=10, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp (s)"], df["gaze_smoothed"], color="red", linewidth=2)
    add_shaded_regions(ax, fruit_timings)
    add_button_press_lines(ax, button_presses)
    ax.set_title(f"Gaze_{filename} (With Smoothing)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vergence (degrees)")
    ax.grid(True)
    fig.savefig(os.path.join(gaze_smooth, f"Gaze_{filename}_with_smoothing.png"))
    plt.close(fig)
