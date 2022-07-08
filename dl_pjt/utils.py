import pandas as pd
import numpy as np

def get_dataset_xy(is_train=True):
    df = __load_dataframe(is_train)
    y = df.iloc[:,:-1]
    x = df.iloc[:,-1]
    x = __convert_image_dataset(x)
    
    return x, y

def __load_dataframe(is_train=True):
    base_path="./data/facial-keypoints-detection/"
    if is_train:
        file_name = "training.csv"
    else:
        file_name = "test.csv"
    return pd.read_csv(base_path + file_name)

def __convert_image_dataset(raw_image_infos):
    image_infos= np.array([list(map(int, raw_img_info.split())) for raw_img_info in raw_image_infos]).reshape(-1,96,96)
    return image_infos

def normalize_image(x):
    x = x.astype(np.float32)/255.0
    return x

def split_data(x, y, train_ratio=0.8):
    row = x.shape[0]
    train_count = int(row * train_ratio)
    valid_count = row - train_count
    indices = np.random.choice(row, row)
    x = [x[:train_count], x[train_count:]]
    y = [y[:train_count], y[train_count:]]
    return x, y

