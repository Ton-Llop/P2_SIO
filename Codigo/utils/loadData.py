import pandas as pd
import numpy as np
import os

def load_dataset(file_path):
    """
    Loads the dataset from the given file path.
    The dataset is expected to be in a wide format (User x Restaurant matrix).
    '99' values are treated as missing values and replaced with NaN.
    """
    # Read CSV, assuming first column is index (User IDs)
    # The file has a header row.
    # Separator is ';'
    df = pd.read_csv(file_path, sep=';', index_col=0)
    
    # Replace 99 with NaN (assuming 99 is the sentinel for missing values)
    df = df.replace(99, np.nan)
    
    # Convert to numeric, just in case
    df = df.apply(pd.to_numeric)
    
    return df

def load_target(file_path):
    """
    Loads the target recommendations file.
    Expected format: User;Restaurant;Rating (no header)
    """
    df = pd.read_csv(file_path, sep=';', header=None, names=['User', 'Restaurant', 'Rating'])
    return df

def save_results(df, file_path):
    """
    Saves the results dataframe to a CSV file.
    Format: User;Restaurant;Rating (no header)
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    df.to_csv(file_path, sep=';', header=False, index=False)
