import os
import xarray as xr
import numpy as np

def save_preprocessed_data(img_paths, img_size=(479, 1059), n_channels=22,
                            x_path="/tmp/preloaded_x.dat", y_path="/tmp/targets.nc"):
    num_files = len(img_paths)
    shape = (num_files,) + img_size + (n_channels,)
    x_memmap = np.memmap(x_path, dtype='float32', mode='w+', shape=shape)

    y_list = []

    for j, path in enumerate(img_paths):
        ds = xr.open_dataset(path, engine="netcdf4")

        input_ds = ds.drop_vars("y_sensor_no2")
        input_array = input_ds.to_array().values
        x_memmap[j] = np.transpose(input_array, (1, 2, 0))

        y_list.append(ds["y_sensor_no2"])

    x_memmap.flush()

    y = xr.concat(y_list, dim="sample")
    y.to_netcdf(y_path)

    print(f"Saved x to {x_path} and y to {y_path}")
