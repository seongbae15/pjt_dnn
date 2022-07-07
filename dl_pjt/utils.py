import pandas as pd
import numpy as np
def load_dataset(is_train=True):
    base_path="./data/facial-keypoints-detection/"
    if is_train:
        file_name = "training.csv"
    else:
        file_name = "test.csv"
    df = pd.read_csv(base_path + file_name)
    y = df.iloc[:,:-1]
    x = df.iloc[:,-1]
    __convert_image_dataset(x)
    return x, y


def __convert_image_dataset(raw_image_infos):
    image_infos = [raw_image_info.split() for raw_image_info in raw_image_infos]
    print(np.array(image_infos).shape)

