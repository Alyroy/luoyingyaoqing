import os
import pandas as pd

def create_directory(directory: str):
    """Creates a directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)
        
        
def load_prompt(file_name:str):
    PROMPT_FOLDER = 'prompt_conf/'
    with open(os.path.join(PROMPT_FOLDER, file_name), 'r') as file:
        return file.read()
    
    
def preprocess_df(f):
    df = pd.read_csv(f, skipinitialspace=True)
    cols = df.columns.tolist()
    for c in cols:
        if "Unnamed:" in c:
            df.drop(columns=c, inplace=True)
    return df


def get_all_files_recursively(directory_path):
    all_files = []
    for foldername, subfolders, filenames in os.walk(directory_path):
        for filename in filenames:
            all_files.append(os.path.join(foldername, filename))
    return all_files