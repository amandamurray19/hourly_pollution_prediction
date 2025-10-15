import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
import multiprocessing as mp
from shared_data_loader import SharedData

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
        inputs = Input(shape=input_shape)
        outputs = layers.Dense(1, activation='linear', use_bias=True)(inputs)
        return models.Model(inputs=inputs, outputs=outputs)

    def __init__(self, sensor_df, output_dir, trial_name, batch_size, epochs,
                 kernel_sizes=[3, 5, 3], filters=64, use_maxpool=False, pool_size=None,
                 activation="relu", upsample_method="nearest", img_size=(479, 1059),
                 data=None):
        self.trial_name = trial_name
        self.sensor_df = sensor_df
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
        self.output_dir = output_dir
        self.data = data

    def run_model(self, CNN=True):
        for cv_group in range(1, 6):
            drop_locs_train_set = self.sensor_df[self.sensor_df["cv_group"].isin([f"{cv_group}", "test"])][["lat", "lon"]]
            drop_locs_val_set = self.sensor_df[~self.sensor_df["cv_group"].isin([f"{cv_group}"])][["lat", "lon"]]

            Y_train = self.data.get_targets_with_mask(drop_locs_train_set)
            Y_val = self.data.get_targets_with_mask(drop_locs_val_set)

            train_loader = tf.data.Dataset.from_tensor_slices((self.data.x, Y_train)).batch(self.batch_size)
            val_loader = tf.data.Dataset.from_tensor_slices((self.data.x, Y_val)).batch(self.batch_size)

            if CNN:
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
                model = self.build_baseline_model(input_shape=self.img_size + (22,))

            model.compile(optimizer="adam", loss=self.masked_mse)
            model.fit(train_loader, validation_data=val_loader, epochs=self.epochs, verbose=1)

def set_gpu(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

def run_experiment(gpu_id, trial_config):
    set_gpu(gpu_id)
    print(f"[GPU {gpu_id}] Starting: {trial_config['trial_name']}")

    shared_data = SharedData("/tmp/preloaded_x.dat", "/tmp/targets.nc", trial_config["num_samples"])

    cv = CrossValidator(
        sensor_df=trial_config['sensor_df'],
        output_dir=".",
        trial_name=trial_config['trial_name'],
        batch_size=trial_config.get("batch_size", 4),
        epochs=trial_config.get("epochs", 5),
        kernel_sizes=trial_config.get("kernel_sizes", [3, 3, 3]),
        filters=trial_config.get("filters", 64),
        use_maxpool=trial_config.get("use_maxpool", False),
        pool_size=trial_config.get("pool_size", (4, 4)),
        activation=trial_config.get("activation", "relu"),
        upsample_method=trial_config.get("upsample_method", "nearest"),
        data=shared_data
    )

    cv.run_model(CNN=trial_config.get("CNN", True))
    print(f"[GPU {gpu_id}] Done: {trial_config['trial_name']}")

def main():
    folder = "../../data/model_data/mini_data_set"
    full_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".nc")]
    sensor_df = pd.read_csv("../../data/sensor_data/final_sensor_cvgroups.csv", index_col=0)

    num_samples = len(full_paths)

    trials = [
        {'trial_name': 'exp_k3_f32', 'kernel_sizes': [3, 3, 3], 'filters': 32, 'use_maxpool': False, 'CNN': True},
        {'trial_name': 'exp_k3_f64', 'kernel_sizes': [3, 3, 3], 'filters': 64, 'use_maxpool': False, 'CNN': True},
        {'trial_name': 'exp_k5_f64', 'kernel_sizes': [5, 5, 5], 'filters': 64, 'use_maxpool': True,  'CNN': True},
        {'trial_name': 'exp_k7_f64', 'kernel_sizes': [7, 7, 7], 'filters': 64, 'use_maxpool': True,  'CNN': True},
        {'trial_name': 'exp_k5_f128','kernel_sizes': [5, 5, 5], 'filters': 128,'use_maxpool': False, 'CNN': True},
        {'trial_name': 'exp_k5_f32', 'kernel_sizes': [5, 5, 5], 'filters': 32, 'use_maxpool': True,  'CNN': True},
        {'trial_name': 'baseline_1', 'CNN': False},
        {'trial_name': 'baseline_2', 'CNN': False},
    ]

    for trial in trials:
        trial['sensor_df'] = sensor_df
        trial['num_samples'] = num_samples

    processes = []
    for i in range(8):
        p = mp.Process(target=run_experiment, args=(i, trials[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All experiments completed.")

if __name__ == "__main__":
    main()
