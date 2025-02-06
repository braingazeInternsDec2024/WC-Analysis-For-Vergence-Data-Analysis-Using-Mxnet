import pandas as pd
import os

def remove_duplicates_from_csv(folder_path):
  """
  Removes duplicate rows from all CSV files in a specified folder.

  Args:
    folder_path: The path to the folder containing the CSV files.
  """

  for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
      filepath = os.path.join(folder_path, filename)
      try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(filepath)
        
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)
        
        # Save the cleaned DataFrame back to the same CSV file
        df.to_csv(filepath, index=False)
        print(f"Removed duplicates from {filename}")
      except Exception as e:
        print(f"Error processing {filename}: {e}")

# Replace 'path/to/your/folder' with the actual path to your folder
folder_path = 'outputs with 100 trials' 
remove_duplicates_from_csv(folder_path)