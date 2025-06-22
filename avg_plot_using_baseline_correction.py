import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


def compute_and_save_average_vergence_for_all_fruits(output_base_dir, smoothing_technique=None):
    for session_folder in os.listdir(output_base_dir):
        session_path = os.path.join(output_base_dir, session_folder)

        if not os.path.isdir(session_path):
            continue

        # Define paths for smoothed and non-smoothed data for direction and gaze arrows
        smoothened_direction_path = os.path.join(session_path, f"smoothened_using_{smoothing_technique}", "direction")
        not_smoothened_direction_path = os.path.join(session_path, "not smoothened", "direction")
        smoothened_gaze_arrows_path = os.path.join(session_path, f"smoothened_using_{smoothing_technique}", "gaze arrows")
        not_smoothened_gaze_arrows_path = os.path.join(session_path, "not smoothened", "gaze arrows")

        # Define paths for storing average plots
        average_plots_path = os.path.join(session_path, "average_plots")
        smoothed_path = os.path.join(average_plots_path, f"smoothed_using_{smoothing_technique}")
        not_smoothed_path = os.path.join(average_plots_path, "not_smoothed")
        baseline_corrected_path = os.path.join(average_plots_path, "baseline_corrected")
        smoothed_baseline_corrected_path = os.path.join(baseline_corrected_path, f"smoothed_using_{smoothing_technique}")
        not_smoothed_baseline_corrected_path = os.path.join(baseline_corrected_path, "not_smoothed")

        os.makedirs(smoothed_path, exist_ok=True)
        os.makedirs(not_smoothed_path, exist_ok=True)
        os.makedirs(baseline_corrected_path, exist_ok=True)
        os.makedirs(smoothed_baseline_corrected_path, exist_ok=True)
        os.makedirs(not_smoothed_baseline_corrected_path, exist_ok=True)

        # Get fruit categories dynamically from the smoothed direction folder
        fruits = os.listdir(smoothened_direction_path) if os.path.exists(smoothened_direction_path) else []

        for fruit in fruits:
            fruit_smoothened_direction_path = os.path.join(smoothened_direction_path, fruit)
            fruit_not_smoothened_direction_path = os.path.join(not_smoothened_direction_path, fruit)
            fruit_smoothened_gaze_arrows_path = os.path.join(smoothened_gaze_arrows_path, fruit)
            fruit_not_smoothened_gaze_arrows_path = os.path.join(not_smoothened_gaze_arrows_path, fruit)

            if not (os.path.isdir(fruit_smoothened_direction_path) and
                    os.path.isdir(fruit_not_smoothened_direction_path) and
                    os.path.isdir(fruit_smoothened_gaze_arrows_path) and
                    os.path.isdir(fruit_not_smoothened_gaze_arrows_path)):
                print(f"Skipping {fruit} in session {session_folder} due to missing data.")
                continue

            # Initialize dictionaries to store summed vergence values and counts for both direction and gaze arrows
            sum_direction_smoothed = {}
            sum_direction_nonsmoothed = {}
            count_direction_smoothed = {}
            count_direction_nonsmoothed = {}

            sum_gaze_arrows_smoothed = {}
            sum_gaze_arrows_nonsmoothed = {}
            count_gaze_arrows_smoothed = {}
            count_gaze_arrows_nonsmoothed = {}

            # Collect CSV files for direction and gaze arrows
            direction_files_smoothed = sorted([f for f in os.listdir(fruit_smoothened_direction_path) if f.endswith(".csv")])
            direction_files_nonsmoothed = sorted([f for f in os.listdir(fruit_not_smoothened_direction_path) if f.endswith(".csv")])
            gaze_arrows_files_smoothed = sorted([f for f in os.listdir(fruit_smoothened_gaze_arrows_path) if f.endswith(".csv")])
            gaze_arrows_files_nonsmoothed = sorted([f for f in os.listdir(fruit_not_smoothened_gaze_arrows_path) if f.endswith(".csv")])

            def extract_event_number(filename):
                match = re.search(r"event_(\d+)", filename)
                return int(match.group(1)) if match else None

            direction_events_smoothed = {extract_event_number(f): f for f in direction_files_smoothed}
            direction_events_nonsmoothed = {extract_event_number(f): f for f in direction_files_nonsmoothed}
            gaze_arrows_events_smoothed = {extract_event_number(f): f for f in gaze_arrows_files_smoothed}
            gaze_arrows_events_nonsmoothed = {extract_event_number(f): f for f in gaze_arrows_files_nonsmoothed}

            common_events_direction = sorted(set(direction_events_smoothed.keys()) & set(direction_events_nonsmoothed.keys()))
            common_events_gaze_arrows = sorted(set(gaze_arrows_events_smoothed.keys()) & set(gaze_arrows_events_nonsmoothed.keys()))

            for event_number in common_events_direction:
                try:
                    df_direction_smoothed = pd.read_csv(os.path.join(fruit_smoothened_direction_path, direction_events_smoothed[event_number]))
                    df_direction_nonsmoothed = pd.read_csv(os.path.join(fruit_not_smoothened_direction_path, direction_events_nonsmoothed[event_number]))

                    vergence_column_direction = "vergence from direction in degrees"

                    # Process direction data (smoothed and non-smoothed)
                    for index, row in df_direction_smoothed.iterrows():
                        frame_number = int(row["frame_number"])
                        vergence_value = row[vergence_column_direction]
                        if not pd.isna(vergence_value):
                            if frame_number not in sum_direction_smoothed:
                                sum_direction_smoothed[frame_number] = 0
                                count_direction_smoothed[frame_number] = 0
                            sum_direction_smoothed[frame_number] += vergence_value
                            count_direction_smoothed[frame_number] += 1

                    for index, row in df_direction_nonsmoothed.iterrows():
                        frame_number = int(row["frame_number"])
                        vergence_value = row[vergence_column_direction]
                        if not pd.isna(vergence_value):
                            if frame_number not in sum_direction_nonsmoothed:
                                sum_direction_nonsmoothed[frame_number] = 0
                                count_direction_nonsmoothed[frame_number] = 0
                            sum_direction_nonsmoothed[frame_number] += vergence_value
                            count_direction_nonsmoothed[frame_number] += 1

                except FileNotFoundError:
                    print(f"Warning: Missing CSV for event {event_number} in {fruit}, skipping this event.")
                    continue

            for event_number in common_events_gaze_arrows:
                try:
                    df_gaze_arrows_smoothed = pd.read_csv(os.path.join(fruit_smoothened_gaze_arrows_path, gaze_arrows_events_smoothed[event_number]))
                    df_gaze_arrows_nonsmoothed = pd.read_csv(os.path.join(fruit_not_smoothened_gaze_arrows_path, gaze_arrows_events_nonsmoothed[event_number]))

                    vergence_column_gaze_arrows = "vergence from gaze arrows in degrees"

                    # Process gaze arrows data (smoothed and non-smoothed)
                    for index, row in df_gaze_arrows_smoothed.iterrows():
                        frame_number = int(row["frame_number"])
                        vergence_value = row[vergence_column_gaze_arrows]
                        if not pd.isna(vergence_value):
                            if frame_number not in sum_gaze_arrows_smoothed:
                                sum_gaze_arrows_smoothed[frame_number] = 0
                                count_gaze_arrows_smoothed[frame_number] = 0
                            sum_gaze_arrows_smoothed[frame_number] += vergence_value
                            count_gaze_arrows_smoothed[frame_number] += 1

                    for index, row in df_gaze_arrows_nonsmoothed.iterrows():
                        frame_number = int(row["frame_number"])
                        vergence_value = row[vergence_column_gaze_arrows]
                        if not pd.isna(vergence_value):
                            if frame_number not in sum_gaze_arrows_nonsmoothed:
                                sum_gaze_arrows_nonsmoothed[frame_number] = 0
                                count_gaze_arrows_nonsmoothed[frame_number] = 0
                            sum_gaze_arrows_nonsmoothed[frame_number] += vergence_value
                            count_gaze_arrows_nonsmoothed[frame_number] += 1

                except FileNotFoundError:
                    print(f"Warning: Missing CSV for event {event_number} in {fruit}, skipping this event.")
                    continue

            # Compute averages for both direction and gaze arrows
            frame_numbers = sorted(sum_direction_smoothed.keys())
            avg_direction_smoothed = [sum_direction_smoothed[f] / count_direction_smoothed[f] for f in frame_numbers]
            avg_direction_nonsmoothed = [sum_direction_nonsmoothed[f] / count_direction_nonsmoothed[f] for f in frame_numbers]

            avg_gaze_arrows_smoothed = [sum_gaze_arrows_smoothed[f] / count_gaze_arrows_smoothed[f] for f in frame_numbers]
            avg_gaze_arrows_nonsmoothed = [sum_gaze_arrows_nonsmoothed[f] / count_gaze_arrows_nonsmoothed[f] for f in frame_numbers]

            # Baseline correction
            baseline_frames = 10
            baseline_direction_smoothed = np.mean(avg_direction_smoothed[:min(baseline_frames, len(avg_direction_smoothed))]) if avg_direction_smoothed else 0
            baseline_direction_nonsmoothed = np.mean(avg_direction_nonsmoothed[:min(baseline_frames, len(avg_direction_nonsmoothed))]) if avg_direction_nonsmoothed else 0
            baseline_gaze_arrows_smoothed = np.mean(avg_gaze_arrows_smoothed[:min(baseline_frames, len(avg_gaze_arrows_smoothed))]) if avg_gaze_arrows_smoothed else 0
            baseline_gaze_arrows_nonsmoothed = np.mean(avg_gaze_arrows_nonsmoothed[:min(baseline_frames, len(avg_gaze_arrows_nonsmoothed))]) if avg_gaze_arrows_nonsmoothed else 0

            avg_direction_smoothed_corrected = [val - baseline_direction_smoothed for val in avg_direction_smoothed]
            avg_direction_nonsmoothed_corrected = [val - baseline_direction_nonsmoothed for val in avg_direction_nonsmoothed]
            avg_gaze_arrows_smoothed_corrected = [val - baseline_gaze_arrows_smoothed for val in avg_gaze_arrows_smoothed]
            avg_gaze_arrows_nonsmoothed_corrected = [val - baseline_gaze_arrows_nonsmoothed for val in avg_gaze_arrows_nonsmoothed]


            # Ensure output directories exist for fruit-specific folders
            smoothed_fruit_folder = os.path.join(smoothed_path, fruit)
            not_smoothed_fruit_folder = os.path.join(not_smoothed_path, fruit)
            smoothed_baseline_corrected_fruit_folder = os.path.join(smoothed_baseline_corrected_path, fruit)
            not_smoothed_baseline_corrected_fruit_folder = os.path.join(not_smoothed_baseline_corrected_path, fruit)

            os.makedirs(smoothed_fruit_folder, exist_ok=True)
            os.makedirs(not_smoothed_fruit_folder, exist_ok=True)
            os.makedirs(smoothed_baseline_corrected_fruit_folder, exist_ok=True)
            os.makedirs(not_smoothed_baseline_corrected_fruit_folder, exist_ok=True)


            # Save CSV for smoothed and non-smoothed and baseline corrected data
            avg_df_smoothed_direction = pd.DataFrame({
                "frame_number": frame_numbers,
                "average vergence from direction (smoothed, degrees)": avg_direction_smoothed
            })
            avg_df_smoothed_gaze_arrows = pd.DataFrame({
                "frame_number": frame_numbers,
                "average vergence from gaze arrows (smoothed, degrees)": avg_gaze_arrows_smoothed
            })
            avg_df_nonsmoothed_direction = pd.DataFrame({
                "frame_number": frame_numbers,
                "average vergence from direction (nonsmoothed, degrees)": avg_direction_nonsmoothed
            })
            avg_df_nonsmoothed_gaze_arrows = pd.DataFrame({
                "frame_number": frame_numbers,
                "average vergence from gaze arrows (nonsmoothed, degrees)": avg_gaze_arrows_nonsmoothed
            })

            avg_df_smoothed_direction_corrected = pd.DataFrame({
                "frame_number": frame_numbers,
                "average vergence from direction (smoothed, baseline corrected, degrees)": avg_direction_smoothed_corrected
            })
            avg_df_smoothed_gaze_arrows_corrected = pd.DataFrame({
                "frame_number": frame_numbers,
                "average vergence from gaze arrows (smoothed, baseline corrected, degrees)": avg_gaze_arrows_smoothed_corrected
            })
            avg_df_nonsmoothed_direction_corrected = pd.DataFrame({
                "frame_number": frame_numbers,
                "average vergence from direction (nonsmoothed, baseline corrected, degrees)": avg_direction_nonsmoothed_corrected
            })
            avg_df_nonsmoothed_gaze_arrows_corrected = pd.DataFrame({
                "frame_number": frame_numbers,
                "average vergence from gaze arrows (nonsmoothed, baseline corrected, degrees)": avg_gaze_arrows_nonsmoothed_corrected
            })


            # Save CSV files
            avg_df_smoothed_direction.to_csv(os.path.join(smoothed_fruit_folder, f"{fruit}_average_direction.csv"), index=False)
            avg_df_smoothed_gaze_arrows.to_csv(os.path.join(smoothed_fruit_folder, f"{fruit}_average_gaze_arrows.csv"), index=False)
            avg_df_nonsmoothed_direction.to_csv(os.path.join(not_smoothed_fruit_folder, f"{fruit}_average_direction.csv"), index=False)
            avg_df_nonsmoothed_gaze_arrows.to_csv(os.path.join(not_smoothed_fruit_folder, f"{fruit}_average_gaze_arrows.csv"), index=False)

            avg_df_smoothed_direction_corrected.to_csv(os.path.join(smoothed_baseline_corrected_fruit_folder, f"{fruit}_average_direction_baseline_corrected.csv"), index=False)
            avg_df_smoothed_gaze_arrows_corrected.to_csv(os.path.join(smoothed_baseline_corrected_fruit_folder, f"{fruit}_average_gaze_arrows_baseline_corrected.csv"), index=False)
            avg_df_nonsmoothed_direction_corrected.to_csv(os.path.join(not_smoothed_baseline_corrected_fruit_folder, f"{fruit}_average_direction_baseline_corrected.csv"), index=False)
            avg_df_nonsmoothed_gaze_arrows_corrected.to_csv(os.path.join(not_smoothed_baseline_corrected_fruit_folder, f"{fruit}_average_gaze_arrows_baseline_corrected.csv"), index=False)


            # Plot and save the results for direction and gaze arrows smoothed and non-smoothed separately and baseline corrected
            if avg_direction_smoothed:
                plt.figure(figsize=(12, 6))
                plt.plot(frame_numbers, avg_direction_smoothed, label=f"Smoothed ({smoothing_technique})", color="blue")
                plt.title(f"Average Vergence from Direction (Smoothed) for {fruit} in {session_folder}")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (degrees)")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(smoothed_fruit_folder, f"average_vergence_{fruit}_smoothed_direction_{session_folder}.png"))
                plt.close()

            if avg_direction_nonsmoothed:
                plt.figure(figsize=(12, 6))
                plt.plot(frame_numbers, avg_direction_nonsmoothed, label="Non-Smoothed (Direction)", color="red")
                plt.title(f"Average Vergence from Direction (Non-Smoothed) for {fruit} in {session_folder}")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (degrees)")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(not_smoothed_fruit_folder, f"average_vergence_{fruit}_nonsmoothed_direction_{session_folder}.png"))
                plt.close()

            if avg_gaze_arrows_smoothed:
                plt.figure(figsize=(12, 6))
                plt.plot(frame_numbers, avg_gaze_arrows_smoothed, label="Smoothed (Gaze Arrows)", color="green")
                plt.title(f"Average Vergence from Gaze Arrows (Smoothed) for {fruit} in {session_folder}")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (degrees)")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(smoothed_fruit_folder, f"average_vergence_{fruit}_smoothed_gaze_arrows_{session_folder}.png"))
                plt.close()

            if avg_gaze_arrows_nonsmoothed:
                plt.figure(figsize=(12, 6))
                plt.plot(frame_numbers, avg_gaze_arrows_nonsmoothed, label="Non-Smoothed (Gaze Arrows)", color="purple")
                plt.title(f"Average Vergence from Gaze Arrows (Non-Smoothed) for {fruit} in {session_folder}")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (degrees)")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(not_smoothed_fruit_folder, f"average_vergence_{fruit}_nonsmoothed_gaze_arrows_{session_folder}.png"))
                plt.close()


            # Plot and save the baseline corrected results
            if avg_direction_smoothed_corrected:
                plt.figure(figsize=(12, 6))
                plt.plot(frame_numbers, avg_direction_smoothed_corrected, label=f"Smoothed ({smoothing_technique}) Baseline Corrected", color="blue")
                plt.axhline(y=0, color='r', linestyle='--') # Add horizontal line at y=0
                plt.title(f"Baseline Corrected Average Vergence from Direction (Smoothed) for {fruit} in {session_folder}")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (Baseline Corrected, degrees)")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(smoothed_baseline_corrected_fruit_folder, f"average_vergence_{fruit}_smoothed_direction_baseline_corrected_{session_folder}.png"))
                plt.close()

            if avg_direction_nonsmoothed_corrected:
                plt.figure(figsize=(12, 6))
                plt.plot(frame_numbers, avg_direction_nonsmoothed_corrected, label="Non-Smoothed (Direction) Baseline Corrected", color="red")
                plt.axhline(y=0, color='r', linestyle='--') # Add horizontal line at y=0
                plt.title(f"Baseline Corrected Average Vergence from Direction (Non-Smoothed) for {fruit} in {session_folder}")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (Baseline Corrected, degrees)")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(not_smoothed_baseline_corrected_fruit_folder, f"average_vergence_{fruit}_nonsmoothed_direction_baseline_corrected_{session_folder}.png"))
                plt.close()

            if avg_gaze_arrows_smoothed_corrected:
                plt.figure(figsize=(12, 6))
                plt.plot(frame_numbers, avg_gaze_arrows_smoothed_corrected, label="Smoothed (Gaze Arrows) Baseline Corrected", color="green")
                plt.axhline(y=0, color='r', linestyle='--') # Add horizontal line at y=0
                plt.title(f"Baseline Corrected Average Vergence from Gaze Arrows (Smoothed) for {fruit} in {session_folder}")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (Baseline Corrected, degrees)")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(smoothed_baseline_corrected_fruit_folder, f"average_vergence_{fruit}_smoothed_gaze_arrows_baseline_corrected_{session_folder}.png"))
                plt.close()

            if avg_gaze_arrows_nonsmoothed_corrected:
                plt.figure(figsize=(12, 6))
                plt.plot(frame_numbers, avg_gaze_arrows_nonsmoothed_corrected, label="Non-Smoothed (Gaze Arrows) Baseline Corrected", color="purple")
                plt.axhline(y=0, color='r', linestyle='--') # Add horizontal line at y=0
                plt.title(f"Baseline Corrected Average Vergence from Gaze Arrows (Non-Smoothed) for {fruit} in {session_folder}")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (Baseline Corrected, degrees)")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(not_smoothed_baseline_corrected_fruit_folder, f"average_vergence_{fruit}_nonsmoothed_gaze_arrows_baseline_corrected_{session_folder}.png"))
                plt.close()


            print(f"Saved average and baseline corrected plots and CSV for {fruit} in {session_folder}")

# Example Usage
output_directory = "output_plots"
compute_and_save_average_vergence_for_all_fruits(output_directory, smoothing_technique="moving_median_avg_filter")

