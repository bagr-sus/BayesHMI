import os
import pickle
import logging

import arviz as az
import numpy as np
import xarray as xr

from ..definitions import ROOT_DIR

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

def idata_from_observe_times(csv_input, mimic_idata, data_index=3):

    posterior_dict = {
        param: (["chain", "draw"], csv_input[:, i + data_index].reshape(1, -1)) for i, param in enumerate(mimic_idata.posterior.data_vars)
    }

    posterior_ds = xr.Dataset(posterior_dict, coords={"chain": [1], "draw": np.arange(csv_input.shape[0])})

    idata = az.InferenceData(posterior=posterior_ds)


    return idata

def thin_inference_data(idata: az.InferenceData, n: int) -> az.InferenceData:
    # Go through each group in the InferenceData
    logging.info(f"Thinning InferenceData by {n}x")
    thinned_groups = {}
    for group_name in idata._groups:
        group = getattr(idata, group_name, None)
        if group is not None:
            # Check if 'draw' is a dimension, then thin along that dimension
            if "draw" in group.dims:
                thinned_group = group.isel(draw=slice(0, None, n))
            else:
                thinned_group = group
            thinned_groups[group_name] = thinned_group

    return az.InferenceData(**thinned_groups)