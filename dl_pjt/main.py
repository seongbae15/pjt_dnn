import tensorflow as tf
from utils import get_dataset_xy
from utils import normalize_image
from utils import split_data
from model import Facial_Kepoints_Detect


def main():
    train_x, train_y = get_dataset_xy(is_train=True)
    train_x = normalize_image(train_x)
    train_x, train_y = split_data(train_x, train_y)
    
    # model = Facial_Kepoints_Detect(input_size=, output_size=)
    # model.summray()
    return

if __name__ == "__main__":
    main()