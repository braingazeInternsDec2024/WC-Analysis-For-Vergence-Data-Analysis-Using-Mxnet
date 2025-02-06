import os
import pandas as pd
import matplotlib.pyplot as plt

def combine_and_plot_fruits(output_base_dir, smoothing_technique=None):
    for session_folder in os.listdir(output_base_dir):
        session_path = os.path.join(output_base_dir, session_folder)

        if not os.path.isdir(session_path):
            continue

        # Define paths for smoothed and non-smoothed data in average_plots
        average_plots_path = os.path.join(session_path, "average_plots")
        smoothened_average_plots_path = os.path.join(average_plots_path, f"smoothed_using_{smoothing_technique}")
        not_smoothed_average_plots_path = os.path.join(average_plots_path, "not_smoothed")

        # Corrected paths - fruit folders are directly inside smoothed/not_smoothed
        smoothened_fruit_path = smoothened_average_plots_path
        not_smoothed_fruit_path = not_smoothed_average_plots_path

        # List of fruits
        fruits = ["grapes", "pear", "orange"]

        # Function to read and plot combined data from CSVs
        def plot_combined(fruit_paths, plot_title, plot_filename):
            plt.figure(figsize=(12, 6))
            lines_plotted = False  # Flag to track if any lines were plotted
            for fruit, fruit_path in fruit_paths.items():
                print(f"Checking path: {fruit_path}") # Debug: Print path
                if os.path.exists(fruit_path):
                    df = pd.read_csv(fruit_path)
                    print(f"DataFrame for {fruit}:") # Debug: Print DataFrame info
                    print(df.head())
                    print(f"DataTypes:\n{df.dtypes}")
                    if not df.empty: # Check if DataFrame is empty
                        plt.plot(df['frame_number'], df.iloc[:, 1], label=f"{fruit.capitalize()}")
                        lines_plotted = True # Set flag if a line is plotted
                    else:
                        print(f"DataFrame for {fruit} is empty. Skipping plot.") # Debug empty DF
                else:
                    print(f"File not found: {fruit_path}") # Debug: File not found

            plt.title(plot_title)
            plt.xlabel("Frame Number")
            plt.ylabel("Average Vergence (degrees)")
            if lines_plotted: # Only call legend if lines were actually plotted
                plt.legend()
            plt.grid(True)
            plt.savefig(plot_filename)
            plt.close()

        # Prepare paths for combined plots - these are now correctly defined above, no need to recreate
        smoothed_path = smoothened_average_plots_path
        not_smoothed_path = not_smoothed_average_plots_path
        os.makedirs(smoothed_path, exist_ok=True) # These should already exist, but ensure they do
        os.makedirs(not_smoothed_path, exist_ok=True) # These should already exist, but ensure they do

        # Combined paths for direction data (Smoothed)
        direction_paths_smoothed = {
            "grapes": os.path.join(smoothened_fruit_path, "grapes", "grapes_average_direction.csv"), # Corrected path
            "pear": os.path.join(smoothened_fruit_path, "pear", "pear_average_direction.csv"), # Corrected path
            "orange": os.path.join(smoothened_fruit_path, "orange", "orange_average_direction.csv") # Corrected path
        }
        # Create combined direction plot (Smoothed)
        plot_combined(direction_paths_smoothed,
                      f"Average Vergence from Direction (Smoothed) - {session_folder}",
                      os.path.join(smoothed_path, f"combined_average_vergence_direction_smoothed_{session_folder}.png"))

        # Combined paths for gaze arrows data (Smoothed)
        gaze_arrows_paths_smoothed = {
            "grapes": os.path.join(smoothened_fruit_path, "grapes", "grapes_average_gaze_arrows.csv"), # Corrected path
            "pear": os.path.join(smoothened_fruit_path, "pear", "pear_average_gaze_arrows.csv"), # Corrected path
            "orange": os.path.join(smoothened_fruit_path, "orange", "orange_average_gaze_arrows.csv") # Corrected path
        }
        # Create combined gaze arrows plot (Smoothed)
        plot_combined(gaze_arrows_paths_smoothed,
                      f"Average Vergence from Gaze Arrows (Smoothed) - {session_folder}",
                      os.path.join(smoothed_path, f"combined_average_vergence_gaze_arrows_smoothed_{session_folder}.png"))

        # Combined paths for direction data (Non-Smoothed)
        direction_paths_not_smoothed = {
            "grapes": os.path.join(not_smoothed_fruit_path, "grapes", "grapes_average_direction.csv"), # Corrected path
            "pear": os.path.join(not_smoothed_fruit_path, "pear", "pear_average_direction.csv"), # Corrected path
            "orange": os.path.join(not_smoothed_fruit_path, "orange", "orange_average_direction.csv") # Corrected path
        }
        # Create combined direction plot (Non-Smoothed)
        plot_combined(direction_paths_not_smoothed,
                      f"Average Vergence from Direction (Non-Smoothed) - {session_folder}",
                      os.path.join(not_smoothed_path, f"combined_average_vergence_direction_nonsmoothed_{session_folder}.png"))

        # Combined paths for gaze arrows data (Non-Smoothed)
        gaze_arrows_paths_not_smoothed = {
            "grapes": os.path.join(not_smoothed_fruit_path, "grapes", "grapes_average_gaze_arrows.csv"), # Corrected path
            "pear": os.path.join(not_smoothed_fruit_path, "pear", "pear_average_gaze_arrows.csv"), # Corrected path
            "orange": os.path.join(not_smoothed_fruit_path, "orange", "orange_average_gaze_arrows.csv") # Corrected path
        }
        # Create combined gaze arrows plot (Non-Smoothed)
        plot_combined(gaze_arrows_paths_not_smoothed,
                      f"Average Vergence from Gaze Arrows (Non-Smoothed) - {session_folder}",
                      os.path.join(not_smoothed_path, f"combined_average_vergence_gaze_arrows_nonsmoothed_{session_folder}.png"))

        print(f"Combined plots for direction and gaze arrows saved for {session_folder}")

# Example Usage
output_directory = "output_plots"
combine_and_plot_fruits(output_directory, smoothing_technique="moving_average")