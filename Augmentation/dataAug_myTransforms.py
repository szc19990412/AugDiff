import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import myTransforms
from pathlib import Path
from tqdm import tqdm


def img_trans(img, img_name, time=0, SAVE=False):

    preprocess1 = myTransforms.HEDJitter(theta=0.05)
    # print(preprocess1)
    preprocess2 = myTransforms.RandomGaussBlur(radius=[0.5, 1.5])
    # print(preprocess2)
    preprocess3 = myTransforms.RandomAffineCV2(alpha=0.1)  # alpha \in [0,0.15]
    # print(preprocess3)
    preprocess4 = myTransforms.RandomElastic(alpha=2, sigma=0.06)
    # print(preprocess4)
    preprocess5 = myTransforms.RandomChoice([myTransforms.RandomHorizontalFlip(p=1),
                                   myTransforms.RandomVerticalFlip(p=1),
                                   myTransforms.AutoRandomRotation()])  # above is for: randomly selecting one for process
    # print(preprocess5)
    preprocess6 = myTransforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5))
    # print(preprocess6)

    # composeimg = preprocess(img)
    HEDJitterimg = preprocess1(img)
    blurimg = preprocess2(img)
    affinecvimg = preprocess3(img)
    elasticimg = preprocess4(img,mask=None)
    Rotationimg = preprocess5(img)
    Colorimg = preprocess6(img)


    root_save = '/data114_2/shaozc/unitopath-public/512'
    Path(root_save+'/HEDJitter_' + str(time)).mkdir(exist_ok=True, parents=True)
    Path(root_save+'/blurimg_' + str(time)).mkdir(exist_ok=True, parents=True)
    Path(root_save+'/affinecvimg_' + str(time)).mkdir(exist_ok=True, parents=True)
    Path(root_save+'/elasticimg_' + str(time)).mkdir(exist_ok=True, parents=True)
    Path(root_save+'/Rotationimg_' + str(time)).mkdir(exist_ok=True, parents=True)
    Path(root_save+'/Colorimg_' + str(time)).mkdir(exist_ok=True, parents=True)
    if SAVE:
        HEDJitterimg.save(root_save+'/HEDJitter_' + str(time) + f'/{img_name}.png')
        blurimg.save(root_save+'/blurimg_' + str(time) + f'/{img_name}.png')
        affinecvimg.save(root_save+'/affinecvimg_' + str(time) + f'/{img_name}.png')
        elasticimg.save(root_save+'/elasticimg_' + str(time) + f'/{img_name}.png')
        Rotationimg.save(root_save+'/Rotationimg_' + str(time) + f'/{img_name}.png')
        Colorimg.save(root_save+'/Colorimg_' + str(time) + f'/{img_name}.png')


class BaseDataset(Dataset):

    def __init__(self, times):
        self.img_dir = '/data114_2/shaozc/unitopath-public/512/images/'
        self.data = list(Path(self.img_dir).glob('*.png')) #all the original images
        self.times = times

        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample_i = self.data[i]
        img = Image.open(sample_i)
        img_trans(img, Path(sample_i).stem, time=self.times, SAVE=True)
        return 0

if __name__ == '__main__':
    
# To augment 10 times, you can choose the number yourself. When constructing the training dataset for diffusion, we only use augmentation once. 

    for i in range(0, 10): 
        train_dataset = BaseDataset(times=i)
        batch_size = 20
        train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    collate_fn=None, 
                                    num_workers=batch_size)

        for step,_ in enumerate(tqdm(train_loader)):
            pass