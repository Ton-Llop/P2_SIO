import sys
import os
import math
import numpy as np
import pandas as pd

# Function to read a CSV file into a DataFrame with the proper format
def read_file(file_path):
  if not os.path.exists(file_path):
    print(f"ERROR! File not found: {file_path}")
    sys.exit(1)
  else:
    return pd.read_csv(file_path, header=None, sep=";", decimal=".", names=['id_user', 'id_restaurant', 'score'])

# Function to check if two DataFrames have the same length
def check_len(df1, df2):
  if len(df1) != len(df2):
    print(f"ERROR! Files have a different length (#rows): {len(df1)} != {len(df2)}")
    sys.exit(1)

# Function to check if two DataFrames have the same values in a given column
def check_differences(df1, df2, parameter):
  for index, (value1, value2) in enumerate(zip(df1[parameter], df2[parameter])):
    if value1 != value2:
      print(f"ERROR! Difference found at row {index + 1}: {value1} != {value2}")
      sys.exit(1)

# Function to check if a value is a valid float
def is_valid_float(value):
  try:
    float_value = float(value)
    return not math.isnan(float_value)
  except (ValueError, TypeError):
    return False

# Function to check if the 'score' column has valid float values
def check_scores(df):
    score_column = df['score']

    # Check if the 'score' column is already of type float64
    if score_column.dtype == 'float64':
      invalid_indices = score_column.isna()
    else:
      invalid_indices = ~score_column.apply(is_valid_float)

    # If any invalid values found, print error message and exit
    if invalid_indices.any():
      invalid_rows = invalid_indices[invalid_indices].index + 1
      print("ERROR! Invalid score value(s) found at row(s):", invalid_rows.values)
      print("Remember to use '.' as the decimal separator.")
      sys.exit(1)


if __name__ == "__main__":

  # Verify that there are two command line arguments
  if len(sys.argv) != 3:
    print("ERROR! Two command line arguments are required.\nExecute the script as follows: python compute_mae.py path/to/target_recommendations.csv path/to/your_predicted_recommendations.csv")
    sys.exit(1)

  # Load the two files into DataFrames
  df_target, df_predicted = read_file(sys.argv[1]), read_file(sys.argv[2])

  # Check for errors in the DataFrames (length, differences in id_user and id_restaurant)
  check_len(df_target, df_predicted)
  check_differences(df_target, df_predicted, 'id_user')
  check_differences(df_target, df_predicted, 'id_restaurant')
  check_scores(df_target)
  check_scores(df_predicted)

  # Compute MAE between two score columns
  target_scores, predicted_scores = df_target['score'], df_predicted['score']
  mae = round(np.mean(np.abs(target_scores - predicted_scores)),3)
  print(f"MAE: {mae}")
  sys.exit(0);
