import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_average_vergence_per_session(output_directory, smoothing_technique):
    """
    Iterates through all session folders inside output_plots/ and generates separate comparison plots
    for smoothed and non-smoothed vergence data for grapes, orange, and pear.
    
    Args:
        output_directory (str): Path to the output_plots/ directory.
        smoothing_technique (str): The technique used for smoothing (e.g., 'moving_average').
    """
    # Iterate through all session folders
    for session_folder in os.listdir(output_directory):
        session_path = os.path.join(output_directory, session_folder)
        if not os.path.isdir(session_path):
            continue  # Skip if it's not a directory
        
        smoothed_path = os.path.join(session_path, "average_plots", f"smoothed_using_{smoothing_technique}")
        not_smoothed_path = os.path.join(session_path, "average_plots", "not_smoothed")
        
        # Check if necessary folders exist
        if not os.path.exists(smoothed_path) or not os.path.exists(not_smoothed_path):
            continue
        
        fruits = ['grapes', 'orange', 'pear']
        smoothed_data = {}
        not_smoothed_data = {}
        
        for fruit in fruits:
            smoothed_csv = os.path.join(smoothed_path, fruit, f"{fruit}_average.csv")
            not_smoothed_csv = os.path.join(not_smoothed_path, fruit, f"{fruit}_average.csv")
            
            if os.path.exists(smoothed_csv):
                df = pd.read_csv(smoothed_csv)
                column_name = f"average vergence from direction (smoothed, degrees)"
                if column_name in df.columns:
                    smoothed_data[fruit] = df
                else:
                    print(f"Warning: Column '{column_name}' not found in {smoothed_csv}")
            
            if os.path.exists(not_smoothed_csv):
                df = pd.read_csv(not_smoothed_csv)
                column_name = f"average vergence from direction (nonsmoothed, degrees)"
                if column_name in df.columns:
                    not_smoothed_data[fruit] = df
                else:
                    print(f"Warning: Column '{column_name}' not found in {not_smoothed_csv}")

        # Plot smoothed data
        if smoothed_data:
            plt.figure(figsize=(10, 6))
            for fruit, df in smoothed_data.items():
                plt.plot(df["frame_number"], df[f"average vergence from direction (smoothed, degrees)"], label=f"{fruit} (Smoothed)")
            plt.xlabel("Frame Number")
            plt.ylabel("Average Vergence (degrees)")
            plt.title(f"Smoothed Vergence - {session_folder}")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(session_path, f"{session_folder}_smoothed_comparison.png"))
            plt.close()
        
        # Plot non-smoothed data
        if not_smoothed_data:
            plt.figure(figsize=(10, 6))
            for fruit, df in not_smoothed_data.items():
                plt.plot(df["frame_number"], df[f"average vergence from direction (nonsmoothed, degrees)"], label=f"{fruit} (Not Smoothed)")
            plt.xlabel("Frame Number")
            plt.ylabel("Average Vergence (degrees)")
            plt.title(f"Not Smoothed Vergence - {session_folder}")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(session_path, f"{session_folder}_not_smoothed_comparison.png"))
            plt.close()

output_directory = "output_plots"
smoothing_technique = "moving_average"
plot_average_vergence_per_session(output_directory, smoothing_technique)
