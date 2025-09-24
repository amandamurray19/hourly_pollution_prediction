import os
import xarray as xr
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from tensorflow.keras import layers, models, Input
import tensorflow as tf
import time


class PreloadedData:
    def __init__(self, img_paths, img_size=(479, 1059), n_channels=22):
        self.img_paths = img_paths
        self.img_size = img_size
        self.n_channels = n_channels

        # Preallocate arrays
        self.x = np.zeros((len(img_paths),) + img_size + (n_channels,), dtype="float32")

        # We'll build y as an xarray Dataset
        y_list = []

        for j, path in enumerate(img_paths):
            ds = xr.open_dataset(path, engine="netcdf4")

            # Input tensor
            input_ds = ds.drop_vars("y_sensor_no2")
            input_array = input_ds.to_array().values
            self.x[j] = np.transpose(input_array, (1, 2, 0))

            # Target tensor
            target_ds = ds["y_sensor_no2"].copy()
            y_list.append(target_ds)

        # Combine all targets into a single Dataset
        self.y = xr.concat(y_list, dim="sample")

    def get_targets_with_mask(self, drop_sensor_locs_df):
        """
        Returns a *new copy* of targets with NaNs applied at drop_sensor_locs.
        """
        target_ds = self.y.copy()

        # Loop over rows to mask each (lat, lon) pair
        for _, row in drop_sensor_locs_df.iterrows():
            target_ds.loc[dict(lat=row['lat'], lon=row['lon'])] = np.nan

        y_masked = target_ds.values
        y_masked = np.expand_dims(y_masked, axis=-1)

        return y_masked


class CrossValidator:
    @staticmethod
    def masked_mse(y_true, y_pred):
        mask = tf.math.logical_not(tf.math.is_nan(y_true))
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)

        diff_squared = tf.square(y_true_masked - y_pred_masked)
        denom = tf.cast(tf.size(diff_squared), tf.float32)
        denom = tf.maximum(denom, 1e-6)
        return tf.reduce_sum(diff_squared) / denom

    @staticmethod
    def build_cnn_model(input_shape, kernel_sizes, filters, activation,
                    use_maxpool=True, pool_size=(2, 2), upsample_method="nearest"):
        inputs = Input(shape=input_shape)
        x = inputs
        for k in kernel_sizes:
            x = layers.Conv2D(filters, (k, k), padding="same")(x)
            x = layers.Activation(activation)(x)
            if use_maxpool:
                x = layers.MaxPooling2D(pool_size=pool_size)(x)
                x = layers.UpSampling2D(size=pool_size, interpolation=upsample_method)(x)
        outputs = layers.Conv2D(1, (kernel_sizes[-1], kernel_sizes[-1]), activation="linear", padding="same")(x)
        return models.Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def build_baseline_model(input_shape=(479, 1059, 22)):
        """
        Build a convolution-free, per-pixel linear regression model.
        Each pixel's 22 input features are combined linearly to produce 1 output value.

        Args:
            input_shape (tuple): Input shape (H, W, C)

        Returns:
            Keras Model
        """
        inputs = Input(shape=input_shape)
        x = inputs

        # Per-pixel linear regression (no hidden layer)
        outputs = layers.Dense(1, activation='linear', use_bias=True)(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def __init__(self, sensor_df, full_paths, output_dir, trial_name, batch_size, epochs, 
                 kernel_sizes=[3, 5, 3], filters=64, use_maxpool=False, pool_size=None,
                 activation="relu", upsample_method="nearest", img_size=(479, 1059)):  
        self.trial_name = trial_name
        self.sensor_df = sensor_df
        self.full_paths = full_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.activation = activation
        self.epochs = epochs
        self.use_maxpool = use_maxpool
        self.pool_size = pool_size
        self.upsample_method = upsample_method
        self.results = []
        self.output_dir = output_dir  # store output directory

        # Preload all data
        print("Preloading all data...")
        self.data = PreloadedData(full_paths, img_size=img_size, n_channels=22)
        print("Done preloading.")

    def run_model(self, CNN=True):
        for cv_group in range(1, 6):
            print(f"Running CV group {cv_group}/6")

            drop_locs_train_set = self.sensor_df[self.sensor_df["cv_group"].isin([f"{cv_group}", "test"])][["lat", "lon"]]
            drop_locs_val_set = self.sensor_df[~self.sensor_df["cv_group"].isin([f"{cv_group}"])][["lat", "lon"]]
            
            # Generate masked targets
            Y_train = self.data.get_targets_with_mask(drop_locs_train_set)
            Y_val = self.data.get_targets_with_mask(drop_locs_val_set)

            # Wrap in Keras Sequence for batching
            train_loader = tf.data.Dataset.from_tensor_slices((self.data.x, Y_train)).batch(self.batch_size)
            val_loader = tf.data.Dataset.from_tensor_slices((self.data.x, Y_val)).batch(self.batch_size)

            with tf.device("/GPU:0"):
                if CNN==True:
                    model = self.build_cnn_model(
                        input_shape=self.img_size + (22,),
                        kernel_sizes=self.kernel_sizes,
                        filters=self.filters,
                        activation=self.activation,
                        use_maxpool=self.use_maxpool,
                        pool_size=self.pool_size,
                        upsample_method=self.upsample_method,
                    )
                else:
                    model = self.build_baseline_model(input_shape=(479, 1059, 22))

                    model.compile(optimizer="adam", loss=self.masked_mse)

                    model.fit(train_loader, validation_data=val_loader, epochs=self.epochs, verbose=1)

            # Predict on full dataset
            y_pred_full = model.predict(self.data.x)
            
            # Collect results
            for j in range(len(self.data.x)):
                y_true = Y_val[j, :, :, 0]  # original target
                y_pred = y_pred_full[j, :, :, 0]  # model predictions

                mask = ~np.isnan(y_true)
                rows, cols = np.where(mask)
                true_vals = y_true[rows, cols]
                pred_vals = y_pred[rows, cols]

                file_name = os.path.basename(self.full_paths[j])

                for r, c, t, p in zip(rows, cols, true_vals, pred_vals):
                    self.results.append({
                        "file": file_name,
                        "row": r,
                        "col": c,
                        "true": t,
                        "predicted": p,
                        "lat": round(r * 0.01 + 28.605, 3),
                        "lon": round(c * 0.01 - 98.895, 3),
                    })

        final_results_all = pd.DataFrame(self.results)
        final_results_all.to_csv(f'all_val_sensors_{self.trial_name}.csv')
        
        y_true = final_results_all['true'].values
        y_pred = final_results_all['predicted'].values
        # Metrics
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mae = np.mean(np.abs(y_true - y_pred))
        # R-squared
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res / ss_tot
        bias = np.mean(y_pred - y_true)
        max_error = np.max(np.abs(y_pred - y_true))
        min_error = np.min(np.abs(y_pred - y_true))
        
        metrics_df = pd.DataFrame({
            "Trial": [self.trial_name],
            "Model": 'CNN',
            "Batch_Size": [self.batch_size],
            "Kernel_Sizes": [str(self.kernel_sizes)],
            "Filters": [self.filters],
            "Activation": [self.activation],
            "Epochs": [self.epochs],
            "Use_MaxPool": [self.use_maxpool],
            "Pool_Size": [str(self.pool_size)],
            "Upsample_Method": [self.upsample_method],
            "RMSE": [rmse],
            "MAE": [mae],
            "R2": [r2],
            "Bias": [bias],
            "Max_Error": [max_error],
            "Min_Error": [min_error]
        })

        metrics_df.to_csv(f'metrics_{self.trial_name}.csv')

        return pd.DataFrame(self.results)

    
    
if __name__ == "__main__":
    
    folder = "../../data/model_data/mini_data_set"
    full_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".nc")][:4]
    sensor_df = pd.read_csv("../../data/sensor_data/final_sensor_cvgroups.csv", index_col=0)
    
    # # Test 1
    cv1 = CrossValidator(
        # Required
        sensor_df=sensor_df,
        full_paths=full_paths,
        output_dir = ".",
        trial_name='CNN_batch4_nopool',
        batch_size=4,
        epochs=5,
        # Rest are optional
        kernel_sizes=[4,4,4],
        filters=64,
        use_maxpool=False,
        pool_size=(4,4),
    )

    cv1.run_model(CNN=True)

    # # Test 2
    cv2 = CrossValidator(
        # Required
        sensor_df=sensor_df,
        output_dir = ".",
        full_paths=full_paths,
        trial_name='CNN_batch4_pool',
        batch_size=4,
        epochs=5,
        # Rest are optional
        kernel_sizes=[4,4,4],
        filters=64,
        use_maxpool=True,
        pool_size=(4,4)
    )
    cv2.run_model(CNN=True)

    # Test 3
    cv3 = CrossValidator(
        # Required
        sensor_df=sensor_df,
        output_dir = ".",
        full_paths=full_paths,
        trial_name='baseline_batch4_pool',
        batch_size=4,
        epochs=5,
        # Rest are optional
    )
    cv3.run_model(CNN=False)