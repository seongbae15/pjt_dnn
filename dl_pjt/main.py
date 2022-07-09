import tensorflow as tf
from utils import get_dataset_xy
from utils import normalize_image
from utils import split_data
from utils import disp_result
from utils import init_GPU_memory
from model import Facial_Kepoints_Detect
from train import train_model


def main():
    init_GPU_memory()
    train_x, train_y = get_dataset_xy(is_train=True)
    train_x = normalize_image(train_x)
    train_x, train_y = split_data(train_x, train_y)
    model = Facial_Kepoints_Detect(
        input_size=[96, 96, 1], output_size=2, init_conv_filters=6,
    )
    batch_size = 32
    lr = 0.001
    epochs = 30
    history = train_model(
        model.get_model(),
        train_x,
        train_y,
        batch_size=batch_size,
        learning_rate=lr,
        epochs=epochs,
    )
    disp_result(history)
    init_GPU_memory()
    return


if __name__ == "__main__":
    main()
