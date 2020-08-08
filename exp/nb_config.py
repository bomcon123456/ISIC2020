
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/config.ipynb

class AppConfig:
    SEED = 8080
    NUM_CLASSES = 7


class PathConfig:
    #     DATA_PATH = '/content/data'
    #     IMAGE_PATH = '/content/data/images'
    #     CSV_PATH = '/content/data/HAM10000_metadata.csv'

    DATA_PATH = "..\data"
    IMAGE_PATH = "..\data\images"
    CSV_PATH = "..\data\HAM10000_metadata.csv"


class TrainConfig:
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-6