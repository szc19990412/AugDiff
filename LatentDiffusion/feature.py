import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path 

class FeatureBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex

class SICAPTrain(FeatureBase):
    def __init__(self, root_dir):
        super().__init__()
        root_dir = root_dir
        augmantation_list = ['images', 'affinecvimg_0', 'blurimg_0', 'Colorimg_0', 'elasticimg_0', 'HEDJitter_0', 'Rotationimg_0']
        augmantation_dict = dict({'images':0, 'affinecvimg_0':1, 'blurimg_0':2, 'Colorimg_0':3, 'elasticimg_0':4, 'HEDJitter_0':5, 'Rotationimg_0':6})

        #---->
        data_buff = []
        data_labels_buff = []
        for aug in augmantation_list:
            data = list(Path(root_dir+aug).glob('*.pt'))
            data_labels = [data[idx].parent.stem for idx in range(len(data))]
            data_buff += data[:int(0.8*(len(data)))]
            data_labels_buff += data_labels[:int(0.8*(len(data)))]
        data_labels_buff = [augmantation_dict[a_label] for a_label in data_labels_buff]
        self.data = ImagePaths(paths=data_buff, labels=dict({'class_label':data_labels_buff}))


class SICAPValidation(FeatureBase):
    def __init__(self, root_dir):
        super().__init__()
        root_dir = root_dir
        augmantation_list = ['images', 'affinecvimg_0', 'blurimg_0', 'Colorimg_0', 'elasticimg_0', 'HEDJitter_0', 'Rotationimg_0']
        augmantation_dict = dict({'images':0, 'affinecvimg_0':1, 'blurimg_0':2, 'Colorimg_0':3, 'elasticimg_0':4, 'HEDJitter_0':5, 'Rotationimg_0':6})

        #---->
        data_buff = []
        data_labels_buff = []
        for aug in augmantation_list:
            data = list(Path(root_dir+aug).glob('*.pt'))
            data_labels = [data[idx].parent.stem for idx in range(len(data))]
            data_buff += data[int(0.8*(len(data))):]
            data_labels_buff += data_labels[int(0.8*(len(data))):]
        data_labels_buff = [augmantation_dict[a_label] for a_label in data_labels_buff]
        self.data = ImagePaths(paths=data_buff, labels=dict({'class_label':data_labels_buff}))

        
class ImagePaths(Dataset):
    def __init__(self, paths, labels=None):

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = [str(path) for path in paths]
        self._length = len(paths)

    def __len__(self):
        return self._length

    def preprocess_pt(self, pt_path):
        pt = torch.load(pt_path)[:, 2:] #remove the coords
        return pt

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_pt(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example