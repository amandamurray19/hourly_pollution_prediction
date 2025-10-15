import numpy as np
import xarray as xr

class SharedData:
    def __init__(self, x_path, y_path, num_samples, img_size=(479, 1059), n_channels=22):
        self.x = np.memmap(x_path, dtype='float32', mode='r', shape=(num_samples,) + img_size + (n_channels,))
        self.y = xr.open_dataset(y_path)["y_sensor_no2"]

    def get_targets_with_mask(self, drop_sensor_locs_df):
        target_ds = self.y.copy()
        for _, row in drop_sensor_locs_df.iterrows():
            target_ds.loc[dict(lat=row['lat'], lon=row['lon'])] = np.nan
        y_masked = target_ds.values
        y_masked = np.expand_dims(y_masked, axis=-1)
        return y_masked
