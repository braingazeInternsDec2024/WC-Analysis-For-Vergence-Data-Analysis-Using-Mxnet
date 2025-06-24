# -*- coding: utf-8 -*-
"""
This script usese BrainGaze class wrapper to process 3 log files given by wctest software
@author: leono
"""

import sys
import numpy as np

from BRAINGAZE_class import *

from VergenceAnalysis import VergenceCalculator
from vergence_plotter_induvidual import plot_all_csv_in_dir


def main(path2logs):

    # Print a start message to confirm the script is running
    print("Starting computations...")

    data = path2logs
    output_folder = "output"

    calculator = VergenceCalculator(data, output_folder)
    calculator.process_videos()

    # plot_all_csv_in_dir(output_folder)


if __name__ == "__main__":
    path2logs = "data"
    print("path2logs= ", path2logs)
    # print('path2logs= ', path2logs)
    main(path2logs)



