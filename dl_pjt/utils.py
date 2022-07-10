import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def get_train_dataset_xy(is_train=True):
    df = __load_dataframe(is_train)
    df = drop_null_columns(df)
    y = df.iloc[:, :-1]
    x = df.iloc[:, -1]
    x = __convert_image_dataset(x)
    return x, y.values


def __load_dataframe(is_train=True):
    base_path = "./data/facial-keypoints-detection/"
    if is_train:
        file_name = "training.csv"
    else:
        file_name = "test.csv"
    return pd.read_csv(base_path + file_name)


def drop_null_columns(df):
    df = df.dropna(axis=1)
    return df


def fill_null_data(df, fill_value):
    df = df.fillna(fill_value)
    return df


def __convert_image_dataset(raw_image_infos):
    image_infos = []
    for raw_img_info in raw_image_infos:
        image_infos.append(list(map(int, raw_img_info.split())))
    image_infos = np.array(image_infos).reshape(-1, 96, 96, 1)
    return image_infos


def normalize_image(x):
    x = x.astype(np.float32) / 255.0
    return x


def split_data(x, y, train_ratio=0.8):
    row = x.shape[0]
    indices = np.random.choice(row, row)
    x = tf.gather(x, indices=indices).numpy()
    y = tf.gather(y, indices=indices).numpy()
    train_count = int(row * train_ratio)
    valid_count = row - train_count
    x0, x1 = tf.split(x, [train_count, valid_count])
    y0, y1 = tf.split(y, [train_count, valid_count])
    x = [x0, x1]
    y = [y0, y1]
    return x, y


def disp_result(history):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], "b-", label="training")
    plt.plot(history.history["val_loss"], "r:", label="validation")
    plt.xlim(5, 30)
    plt.ylim(-5, 15)
    plt.title("model - loss")
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], "b-", label="training")
    plt.plot(history.history["val_accuracy"], "r:", label="validation")
    plt.title("model - accuracy")
    plt.legend()
    plt.tight_layout()
    plt.xlim(5, 30)
    plt.ylim(-5, 15)
    plt.show()
    return


def init_GPU_memory():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
