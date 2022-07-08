import tensorflow
from utils import get_dataset_xy
from utils import split_train_valid


def main():
    train_x, train_y = get_dataset_xy(is_train=True)
    #split train valid
    split_train_valid(train_x, train_y)
    return

if __name__ == "__main__":
    main()