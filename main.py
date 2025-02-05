# -*- coding: utf-8 -*-
"""
This script uses the BrainGaze class wrapper to process log files given by wctest software.
@author: leono
"""

import sys
from BRAINGAZE_class import *
from VergenceAnalysis import VergenceCalculator
from vergence_plotter_induvidual import plot_all_csv_in_dir

def main(path2logs):
    # Print a start message to confirm the script is running
    print(f"Starting computations for path: {path2logs}")

    output_folder = "output"
    
    calculator = VergenceCalculator(path2logs, output_folder)
    calculator.process_videos()

    plot_all_csv_in_dir(output_folder)

if __name__ == "__main__":
    # Check if a command-line argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_logs>")
        sys.exit(1)

    path2logs = sys.argv[1]  # Get the path from command-line argument
    main(path2logs)
