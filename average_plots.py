import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def compute_and_save_average_vergence_for_all_fruits(output_base_dir, smoothing_technique=None):
    """
    Computes and saves the average vergence values for each fruit in each session
    for both smoothed and non-smoothed data (direction and gaze arrows).
    
    Args:
        output_base_dir (str): Path to the base directory containing session data.
        smoothing_technique (str): The technique used for smoothing, if any (e.g., 'moving_average').
    """
    
    # Iterate over all session folders inside output_plots
    for session_folder in os.listdir(output_base_dir):
        session_path = os.path.join(output_base_dir, session_folder)
        
        # Ensure we only process directories
        if not os.path.isdir(session_path):
            continue
        
        # Define paths for smoothed and non-smoothed data with the smoothing technique
        smoothened_direction_path = os.path.join(session_path, f"smoothened_using_{smoothing_technique}", "direction")
        smoothened_gaze_arrows_path = os.path.join(session_path, f"smoothened_using_{smoothing_technique}", "gaze_arrows")
        not_smoothened_direction_path = os.path.join(session_path, "not smoothened", "direction")
        not_smoothened_gaze_arrows_path = os.path.join(session_path, "not smoothened", "gaze_arrows")

        # Directory for saving average vergence data with the smoothing technique in the name
        average_plots_path = os.path.join(session_path, "average_plots")
        smoothed_path = os.path.join(average_plots_path, f"smoothed_using_{smoothing_technique}")  # Updated folder name
        not_smoothed_path = os.path.join(average_plots_path, "not_smoothed")

        os.makedirs(smoothed_path, exist_ok=True)
        os.makedirs(not_smoothed_path, exist_ok=True)

        
        os.makedirs(smoothed_path, exist_ok=True)
        os.makedirs(not_smoothed_path, exist_ok=True)
        
        # Process each fruit (grapes, pears, oranges, etc.) in the session
        for fruit in os.listdir(smoothened_direction_path):
            # Paths for current fruit
            fruit_smoothened_direction_path = os.path.join(smoothened_direction_path, fruit)
            fruit_smoothened_gaze_arrows_path = os.path.join(smoothened_gaze_arrows_path, fruit)
            fruit_not_smoothened_direction_path = os.path.join(not_smoothened_direction_path, fruit)
            fruit_not_smoothened_gaze_arrows_path = os.path.join(not_smoothened_gaze_arrows_path, fruit)
            
            if not os.path.isdir(fruit_smoothened_direction_path) or not os.path.isdir(fruit_smoothened_gaze_arrows_path) or \
               not os.path.isdir(fruit_not_smoothened_direction_path) or not os.path.isdir(fruit_not_smoothened_gaze_arrows_path):
                continue

            all_vergence_values_direction_smoothed = []
            all_vergence_values_gaze_arrows_smoothed = []
            all_vergence_values_direction_nonsmoothed = []
            all_vergence_values_gaze_arrows_nonsmoothed = []
            frame_numbers = None

            # Collect vergence data from each CSV file for the specific fruit (for all four cases)
            for file in sorted(os.listdir(fruit_smoothened_direction_path)):
                if file.endswith(".csv"):
                    # Load the data from all four directories
                    df_direction_smoothed = pd.read_csv(os.path.join(fruit_smoothened_direction_path, file))
                    df_gaze_arrows_smoothed = pd.read_csv(os.path.join(fruit_smoothened_gaze_arrows_path, file))
                    df_direction_nonsmoothed = pd.read_csv(os.path.join(fruit_not_smoothened_direction_path, file))
                    df_gaze_arrows_nonsmoothed = pd.read_csv(os.path.join(fruit_not_smoothened_gaze_arrows_path, file))
                    
                    if frame_numbers is None:
                        frame_numbers = df_direction_smoothed["frame_number"].values  # Assuming same frame numbers for all files
                    
                    # Column names for vergence
                    vergence_column_direction = "vergence from direction in degrees"
                    vergence_column_gaze_arrows = "vergence from gaze arrows in degrees"
                    
                    # Append data for smoothed and non-smoothed cases
                    if vergence_column_direction in df_direction_smoothed.columns:
                        all_vergence_values_direction_smoothed.append(df_direction_smoothed[vergence_column_direction].values)
                    if vergence_column_gaze_arrows in df_gaze_arrows_smoothed.columns:
                        all_vergence_values_gaze_arrows_smoothed.append(df_gaze_arrows_smoothed[vergence_column_gaze_arrows].values)
                    if vergence_column_direction in df_direction_nonsmoothed.columns:
                        all_vergence_values_direction_nonsmoothed.append(df_direction_nonsmoothed[vergence_column_direction].values)
                    if vergence_column_gaze_arrows in df_gaze_arrows_nonsmoothed.columns:
                        all_vergence_values_gaze_arrows_nonsmoothed.append(df_gaze_arrows_nonsmoothed[vergence_column_gaze_arrows].values)

            # Interpolation for frame number alignment if needed
            def align_frames(original_frames, target_frames, data):
                """ Interpolate data to match the target frame numbers """
                interp_func = interp1d(original_frames, data, kind='linear', fill_value="extrapolate")
                return interp_func(target_frames)
            
            # Compute ensemble average for all four categories if data exists
            if (all_vergence_values_direction_smoothed and all_vergence_values_gaze_arrows_smoothed and
                all_vergence_values_direction_nonsmoothed and all_vergence_values_gaze_arrows_nonsmoothed):
                
                # Align frames between smoothed and non-smoothed data
                aligned_direction_smoothed = np.mean([align_frames(df_direction_smoothed["frame_number"], frame_numbers, data) 
                                                      for data in all_vergence_values_direction_smoothed], axis=0)
                aligned_gaze_arrows_smoothed = np.mean([align_frames(df_gaze_arrows_smoothed["frame_number"], frame_numbers, data) 
                                                       for data in all_vergence_values_gaze_arrows_smoothed], axis=0)
                aligned_direction_nonsmoothed = np.mean([align_frames(df_direction_nonsmoothed["frame_number"], frame_numbers, data) 
                                                          for data in all_vergence_values_direction_nonsmoothed], axis=0)
                aligned_gaze_arrows_nonsmoothed = np.mean([align_frames(df_gaze_arrows_nonsmoothed["frame_number"], frame_numbers, data) 
                                                           for data in all_vergence_values_gaze_arrows_nonsmoothed], axis=0)
                
                # Create a separate folder for the current fruit under the smoothed and not_smoothed directories
                smoothed_fruit_folder = os.path.join(smoothed_path, fruit)
                not_smoothed_fruit_folder = os.path.join(not_smoothed_path, fruit)
                os.makedirs(smoothed_fruit_folder, exist_ok=True)
                os.makedirs(not_smoothed_fruit_folder, exist_ok=True)
                
                # Save the averaged vergence data as CSV for smoothed and non-smoothed
                avg_df_smoothed = pd.DataFrame({
                    "frame_number": frame_numbers, 
                    "average vergence from direction (smoothed, degrees)": aligned_direction_smoothed,
                    "average vergence from gaze arrows (smoothed, degrees)": aligned_gaze_arrows_smoothed
                })
                avg_csv_smoothed_path = os.path.join(smoothed_fruit_folder, f"{fruit}_average.csv")
                avg_df_smoothed.to_csv(avg_csv_smoothed_path, index=False)
                
                avg_df_nonsmoothed = pd.DataFrame({
                    "frame_number": frame_numbers, 
                    "average vergence from direction (nonsmoothed, degrees)": aligned_direction_nonsmoothed,
                    "average vergence from gaze arrows (nonsmoothed, degrees)": aligned_gaze_arrows_nonsmoothed
                })
                avg_csv_nonsmoothed_path = os.path.join(not_smoothed_fruit_folder, f"{fruit}_average.csv")
                avg_df_nonsmoothed.to_csv(avg_csv_nonsmoothed_path, index=False)
                
                # Combined Plot (Smoothed)
                plt.figure(figsize=(12, 6))
                plt.plot(avg_df_smoothed["frame_number"], avg_df_smoothed["average vergence from direction (smoothed, degrees)"], label=f"{fruit.capitalize()} Direction Smoothed", color="blue")
                plt.plot(avg_df_smoothed["frame_number"], avg_df_smoothed["average vergence from gaze arrows (smoothed, degrees)"], label=f"{fruit.capitalize()} Gaze Arrows Smoothed", color="green")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (degrees)")
                plt.title(f"Average Vergence Plot - {fruit.capitalize()} (Smoothed)")
                plt.legend()
                plt.grid(True)
                avg_plot_smoothed_path = os.path.join(smoothed_fruit_folder, f"{fruit}_average_combined.png")
                plt.savefig(avg_plot_smoothed_path)
                plt.close()

                # Plot for Direction (Smoothed)
                plt.figure(figsize=(12, 6))
                plt.plot(avg_df_smoothed["frame_number"], avg_df_smoothed["average vergence from direction (smoothed, degrees)"], label=f"{fruit.capitalize()} Direction Smoothed", color="blue")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (degrees)")
                plt.title(f"Direction Vergence - {fruit.capitalize()} (Smoothed)")
                plt.legend()
                plt.grid(True)
                avg_plot_direction_smoothed_path = os.path.join(smoothed_fruit_folder, f"{fruit}_direction_smoothed.png")
                plt.savefig(avg_plot_direction_smoothed_path)
                plt.close()

                # Plot for Gaze Arrows (Smoothed)
                plt.figure(figsize=(12, 6))
                plt.plot(avg_df_smoothed["frame_number"], avg_df_smoothed["average vergence from gaze arrows (smoothed, degrees)"], label=f"{fruit.capitalize()} Gaze Arrows Smoothed", color="green")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (degrees)")
                plt.title(f"Gaze Arrows Vergence - {fruit.capitalize()} (Smoothed)")
                plt.legend()
                plt.grid(True)
                avg_plot_gaze_arrows_smoothed_path = os.path.join(smoothed_fruit_folder, f"{fruit}_gaze_arrows_smoothed.png")
                plt.savefig(avg_plot_gaze_arrows_smoothed_path)
                plt.close()

                # Same for non-smoothed
                # Combined Plot (Non-Smooth)
                plt.figure(figsize=(12, 6))
                plt.plot(avg_df_nonsmoothed["frame_number"], avg_df_nonsmoothed["average vergence from direction (nonsmoothed, degrees)"], label=f"{fruit.capitalize()} Direction Nonsmoothed", color="red")
                plt.plot(avg_df_nonsmoothed["frame_number"], avg_df_nonsmoothed["average vergence from gaze arrows (nonsmoothed, degrees)"], label=f"{fruit.capitalize()} Gaze Arrows Nonsmoothed", color="orange")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (degrees)")
                plt.title(f"Average Vergence Plot - {fruit.capitalize()} (Not Smoothed)")
                plt.legend()
                plt.grid(True)
                avg_plot_nonsmoothed_path = os.path.join(not_smoothed_fruit_folder, f"{fruit}_average_combined.png")
                plt.savefig(avg_plot_nonsmoothed_path)
                plt.close()

                # Direction Plot (Non-Smooth)
                plt.figure(figsize=(12, 6))
                plt.plot(avg_df_nonsmoothed["frame_number"], avg_df_nonsmoothed["average vergence from direction (nonsmoothed, degrees)"], label=f"{fruit.capitalize()} Direction Nonsmoothed", color="red")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (degrees)")
                plt.title(f"Direction Vergence - {fruit.capitalize()} (Not Smoothed)")
                plt.legend()
                plt.grid(True)
                avg_plot_direction_nonsmoothed_path = os.path.join(not_smoothed_fruit_folder, f"{fruit}_direction_nonsmoothed.png")
                plt.savefig(avg_plot_direction_nonsmoothed_path)
                plt.close()

                # Gaze Arrows Plot (Non-Smooth)
                plt.figure(figsize=(12, 6))
                plt.plot(avg_df_nonsmoothed["frame_number"], avg_df_nonsmoothed["average vergence from gaze arrows (nonsmoothed, degrees)"], label=f"{fruit.capitalize()} Gaze Arrows Nonsmoothed", color="orange")
                plt.xlabel("Frame Number")
                plt.ylabel("Average Vergence (degrees)")
                plt.title(f"Gaze Arrows Vergence - {fruit.capitalize()} (Not Smoothed)")
                plt.legend()
                plt.grid(True)
                avg_plot_gaze_arrows_nonsmoothed_path = os.path.join(not_smoothed_fruit_folder, f"{fruit}_gaze_arrows_nonsmoothed.png")
                plt.savefig(avg_plot_gaze_arrows_nonsmoothed_path)
                plt.close()

# Example Usage
output_directory = "output_plots"
compute_and_save_average_vergence_for_all_fruits(output_directory, smoothing_technique="moving_average")
