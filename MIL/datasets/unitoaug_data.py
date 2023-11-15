import random
import torch
import pandas as pd
from pathlib import Path
import glob
import numpy as np

import torch.utils.data as data
from torch.utils.data import dataloader


class UnitoaugData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        #---->state
        self.type = ['images', 'affinecvimg_0', 'blurimg_0', 'Colorimg_0', 'elasticimg_0', 'HEDJitter_0', 'Rotationimg_0']
        self.state = state

        #---->split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train_image_id'].dropna().tolist()*len(self.type) # larger training dataset, each augmented image has the choice to be selected
            self.label = self.slide_data.loc[:, 'train_type_label'].dropna().tolist()*len(self.type)
        if state == 'val':
            self.data = self.slide_data.loc[:, 'val_image_id'].dropna().tolist()*len(self.type)
            self.label = self.slide_data.loc[:, 'val_type_label'].dropna().tolist()*len(self.type)
        if state == 'test':
            self.data = self.slide_data.loc[:, 'test_image_id'].dropna().tolist()
            self.label = self.slide_data.loc[:, 'test_type_label'].dropna().tolist()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = int(self.label[idx])

        #----> test
        if self.state == 'test':
            full_path = Path(self.feature_dir) / 'images' / f'{slide_id}.ndpi.pt' #TODO
            features = torch.load(full_path)    

        else:
            select_type = np.random.choice(self.type)
            full_path = Path(self.feature_dir) / select_type / f'{slide_id}.ndpi.pt' #TODO
            features = torch.load(full_path)



        return features, label

