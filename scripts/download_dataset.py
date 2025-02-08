#!/usr/bin/env python
import argparse
import os
import shutil
import subprocess
import sys

from decouple import config

# Load environment variables
TRAIN_DATASET_ID = config("TRAIN_DATASET_ID")
TEST_DATASET_ID = config("TEST_DATASET_ID")

if TRAIN_DATASET_ID is None or TEST_DATASET_ID is None:
    raise ValueError("TRAIN_DATASET_ID or TEST_DATASET_ID not found in .env file")

# Define dataset IDs
DATASET_IDS = {
    "PS3C-train": TRAIN_DATASET_ID,
    "PS3C-test": TEST_DATASET_ID,
}

def download_and_extract(file_id, destination_folder, filename):
    """
    Download a dataset from Google Drive using gdown and extract it.
    Args:
        file_id (str): The ID of the file to download
        destination_folder (str): The folder to download and extract the dataset to
        filename (str): The name of the .7z file
    """
    os.makedirs(destination_folder, exist_ok=True)
    
    # Download the file using gdown
    print(f"Downloading {filename}...")
    subprocess.run([
        "gdown",
        f"https://drive.google.com/uc?id={file_id}",
        "-O",
        f"{destination_folder}/{filename}"
    ])
    
    # Extract the dataset
    print(f"Extracting {filename}...")
    subprocess.run(["7z", "x", f"{destination_folder}/{filename}", f"-o{destination_folder}"])
    
    # Remove the downloaded .7z file
    os.remove(f"{destination_folder}/{filename}")
    print(f"{filename} downloaded, extracted, and .7z file removed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Download PS3C train and test datasets from Google Drive")
    parser.add_argument(
        "-f", "--folder",
        type=str,
        default="../data",
        help="Folder to download and extract the dataset to"
    )
    args = parser.parse_args()
    
    # Check if gdown is installed
    if not shutil.which("gdown"):
        print("gdown is not installed. Installing gdown...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"])
    
    # Check if 7z is installed
    if not shutil.which("7z"):
        print("7z is not installed. Please install p7zip-full package.")
        sys.exit(1)
    
    # Download and extract train and test datasets
    download_and_extract(DATASET_IDS["PS3C-train"], args.folder, "PS3C_train.7z")
    download_and_extract(DATASET_IDS["PS3C-test"], args.folder, "PS3C_test.7z")

if __name__ == "__main__":
    main()

