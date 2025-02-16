import os
import pathlib
import pickle

import arviz as az

from definitions import ROOT_DIR

def idata_path() -> None:
    return os.path.join(ROOT_DIR, "data", "idata")

def save_idata_to_file(
        idata: az.InferenceData,
        filename: str,
        folder_path: str = idata_path()) -> None:
    # if path doesn't exist, create it
    print(f"Saving idata {filename} to {folder_path}...")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    path = os.path.join(folder_path, filename)

    if os.path.exists(path=path):
        with open(path, "wb") as file:
            pickle.dump(obj=idata, file=file)
    else:
        with open(path, "ab") as file:
            pickle.dump(obj=idata, file=file)

def read_idata_from_file(
        filename: str,
        folder_path: str = idata_path()) -> az.InferenceData:
    path = os.path.join(folder_path, filename)
    print(f"Reading idata from {path}")
    try:
        with open(path, "rb") as file:
            idata = pickle.load(file=file)
            return idata
    except:
        print("Error reading idata file")

