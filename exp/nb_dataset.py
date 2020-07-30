
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/dataset.ipynb

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from exp.nb_config import *

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

def preprocess_df(df, valid_size=0.2, seed=AppConfig.SEED):

    df['path'] = PathConfig.IMAGE_PATH + '/' + df['image_id'] + '.jpg'
    df['label_fullstr'] = df['dx'].map(lesion_type_dict.get)

    label_str = pd.Categorical(df['label_fullstr'])
    df['label_index'] = label_str.codes

    df_undup = df.groupby('lesion_id').count()
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    _, valid = train_test_split(df_undup['lesion_id'], test_size=valid_size,
                                random_state=seed,
                                stratify=df_undup['label_index'])
    valid = set(valid)
    df['val'] = df['lesion_id'].apply(lambda x: 1 if str(x) in valid else 0)

    df_train = df[df['val'] == 0]
    df_valid = df[df['val'] == 1]

    return df_train.reset_index(drop=True), df_valid.reset_index(drop=True), list(label_str.categories)

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __getitem__(self, i):
        x = Image.open(self.df['path'][i])
        y = torch.tensor(int(self.df['label_index'][i]))

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.df)

class SkinDataBunch:
    def __init__(self, train_dl, valid_dl, labels):
        self.train_dl,self.valid_dl,self.labels, self.c = train_dl,valid_dl,labels,len(labels)

    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def valid_ds(self): return self.valid_dl.dataset

    def show_image(self, index=None):
        dataset = self.train_dl.dataset
        n_samples = len(dataset)

        if not index:
            index = int(np.random.random()*n_samples)
        else:
            if index >= n_samples or index < 0:
                print('Invalid index.')
                return

        x, y = dataset[index]

        plt.imshow(x.permute(1,2,0))
        plt.axis('off')
        plt.title(self.labels[y])

    def show_grid(self, n_rows=5, n_cols=5):
        dataset = self.train_dl.dataset
        array = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), n_rows*n_cols, replace=False))

        plt.figure(figsize=(12, 12))
        for row in range(n_rows):
            for col in range(n_cols):
                index = n_cols * row + col
                plt.subplot(n_rows, n_cols, index + 1)
                plt.imshow(array[index][0].permute(1, 2, 0))
                plt.axis('off')
                label = self.labels[int(array[index][1])]
                plt.title(label, fontsize=12)
        plt.tight_layout()
