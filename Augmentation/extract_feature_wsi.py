# -*- coding: utf-8 -*-

import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from torchvision import models
from PIL import ImageFile, Image
import pandas as pd
import glob 
import timm
import pickle
import random
from pathlib import Path
import torch.nn.functional as F
from collections import OrderedDict
from resnet_custom import resnet50_baseline

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#---->
class Dataset_All_Bags(data.Dataset):
    def __init__(self, wsi_path):
        self.image_all = sorted(glob.glob(wsi_path + '*'))
        self.slide = [Path(image).stem.split('_')[0] for image in self.image_all]
        self.slide_unique = np.unique(self.slide)
    
    def __len__(self):
        return len(self.slide_unique)

    def __getitem__(self, idx):
        select_slide = self.slide_unique[idx]
        slide_idx = np.where(np.array(self.slide)==select_slide)[0].tolist()
        return [self.image_all[s_idx] for s_idx in slide_idx]

#---->
class Whole_Slide_Bag(data.Dataset):
    def __init__(self, wsi_path, transform=None):
        self.patch = wsi_path
        self.transform = transform
    
    def __len__(self):
        return len(self.patch)

    def __getitem__(self, idx):

        img = Image.open(self.patch[idx]).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img) 

        #---->coords
        try:
            _, x, _, y = Path(self.patch[idx]).stem.split('/')[-1].split('_')[-4:]
            coord = [int(x), int(y)]
        except:
            coord = [0, 0]

        return img, coord, self.patch[idx]

def collate_features(batch):
    img = torch.stack([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    patch_dir = np.stack([item[2] for item in batch])
    return [img, coords, patch_dir]

# CUDA_VISIBLE_DEVICES=4 python extract_feature_wsi.py --wsi_files=/data114_1/shaozc/SICAPv2/SICAPv2/images/ --output_files='/data114_1/shaozc/SICAPv2/SICAPv2/pt_files_384_wsi/images/'
# CUDA_VISIBLE_DEVICES=3 python extract_feature_wsi.py --wsi_files=/data114_1/shaozc/SICAPv2/SICAPv2/multiple_augmentation/affinecvimg_0/ --output_files='/data114_1/shaozc/SICAPv2/SICAPv2/multiple_augmentation/pt_files_wsi/affinecvimg_0/'
# CUDA_VISIBLE_DEVICES=3 python extract_feature_wsi.py --wsi_files=/data114_1/shaozc/SICAPv2/SICAPv2/multiple_augmentation/blurimg_0/ --output_files='/data114_1/shaozc/SICAPv2/SICAPv2/multiple_augmentation/pt_files_wsi/blurimg_0/'
# CUDA_VISIBLE_DEVICES=3 python extract_feature_wsi.py --wsi_files=/data114_1/shaozc/SICAPv2/SICAPv2/multiple_augmentation/Colorimg_0/ --output_files='/data114_1/shaozc/SICAPv2/SICAPv2/multiple_augmentation/pt_files_wsi/Colorimg_0/'
# CUDA_VISIBLE_DEVICES=3 python extract_feature_wsi.py --wsi_files=/data114_1/shaozc/SICAPv2/SICAPv2/multiple_augmentation/elasticimg_0/ --output_files='/data114_1/shaozc/SICAPv2/SICAPv2/multiple_augmentation/pt_files_wsi/elasticimg_0/'
# CUDA_VISIBLE_DEVICES=3 python extract_feature_wsi.py --wsi_files=/data114_1/shaozc/SICAPv2/SICAPv2/multiple_augmentation/HEDJitter_0/ --output_files='/data114_1/shaozc/SICAPv2/SICAPv2/multiple_augmentation/pt_files_wsi/HEDJitter_0/'
# CUDA_VISIBLE_DEVICES=3 python extract_feature_wsi.py --wsi_files=/data114_1/shaozc/SICAPv2/SICAPv2/multiple_augmentation/Rotationimg_0/ --output_files='/data114_1/shaozc/SICAPv2/SICAPv2/multiple_augmentation/pt_files_wsi/Rotationimg_0/'


# CUDA_VISIBLE_DEVICES=4 python extract_feature_wsi.py --wsi_files=/data114_2/shaozc/unitopath-public/512/affinecvimg_0/ --output_files='/data114_2/shaozc/unitopath-public/512/pt_files_384_wsi/affinecvimg_0/'
# CUDA_VISIBLE_DEVICES=4 python extract_feature_wsi.py --wsi_files=/data114_2/shaozc/unitopath-public/512/blurimg_0/ --output_files='/data114_2/shaozc/unitopath-public/512/pt_files_384_wsi/blurimg_0/'
# CUDA_VISIBLE_DEVICES=4 python extract_feature_wsi.py --wsi_files=/data114_2/shaozc/unitopath-public/512/Colorimg_0/ --output_files='/data114_2/shaozc/unitopath-public/512/pt_files_384_wsi/Colorimg_0/'
# CUDA_VISIBLE_DEVICES=4 python extract_feature_wsi.py --wsi_files=/data114_2/shaozc/unitopath-public/512/elasticimg_0/ --output_files='/data114_2/shaozc/unitopath-public/512/pt_files_384_wsi/elasticimg_0/'
# CUDA_VISIBLE_DEVICES=4 python extract_feature_wsi.py --wsi_files=/data114_2/shaozc/unitopath-public/512/HEDJitter_0/ --output_files='/data114_2/shaozc/unitopath-public/512/pt_files_384_wsi/HEDJitter_0/'
# CUDA_VISIBLE_DEVICES=4 python extract_feature_wsi.py --wsi_files=/data114_2/shaozc/unitopath-public/512/Rotationimg_0/ --output_files='/data114_2/shaozc/unitopath-public/512/pt_files_384_wsi/Rotationimg_0/'
# CUDA_VISIBLE_DEVICES=4 python extract_feature_wsi.py --wsi_files=/data114_2/shaozc/unitopath-public/512/images/ --output_files='/data114_2/shaozc/unitopath-public/512/pt_files_384_wsi/images/'


ImageFile.LOAD_TRUNCATED_IMAGES = True
parser = argparse.ArgumentParser(description='RCC prediction')
parser.add_argument('--wsi_files', type=str, default='/data114_1/shaozc/SICAPv2/SICAPv2/images/')
parser.add_argument('--output_files', default='/data114_1/shaozc/SICAPv2/SICAPv2/pt_files_simsiam_wsi/images/', type=str)
parser.add_argument('--file_path_base', type=str, help='model file path', default='result/resnet18-5c106cde.pth')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--feature_size', default=512, type=int, help='')
# Miscs
parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.gpu:
    torch.cuda.manual_seed_all(args.manualSeed)

if __name__ == '__main__':

    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    normTransform = transforms.Normalize(normMean, normStd)
    test_transform = transforms.Compose([
            # transforms.Resize(224),
            transforms.ToTensor(),
            normTransform
        ])




    # # ResNet
    import timm
    base_model = timm.create_model('resnet18', pretrained=True, num_classes=0) 
    base_model = base_model.to(device)
    # import timm
    # base_model = timm.create_model('regnetx_004', pretrained=True, num_classes=0) 
    # base_model = base_model.to(device)



    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()
    base_model.eval()
    base_model = base_model.to(device)
    # classifier.eval()

    #---->
    bags_dataset = Dataset_All_Bags(args.wsi_files)
    total = len(bags_dataset)


    #---->
    save_dir = Path(args.output_files)
    save_dir.mkdir(exist_ok=True, parents=True)
    dest_files = glob.glob(f'{save_dir}/*')
    # dest_files = [dest[:-5]+'.pt' for dest in dest_files]


    for bag_candidate_idx in range(total):
        slide_patches = bags_dataset[bag_candidate_idx]
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))

        slide_id = Path(slide_patches[0]).stem.split('_')[0]
        if f'{save_dir}/{slide_id}.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue 


        test_data = Whole_Slide_Bag(slide_patches, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=32, shuffle=False,
                num_workers=8, pin_memory=True, collate_fn=collate_features)

        wsi_feature = []
        with torch.no_grad():
            for batch_idx, (inputs, coord, patch_dir) in enumerate(tqdm(test_loader)):
                inputs = inputs.to(device, non_blocking=True)
                features = base_model(inputs)
                # logits = classifier(features)
                # output = F.softmax(logits, dim=1)
                # pseudo_label = torch.max(output, dim=1)[1]
                # #---->cpu
                features = features.cpu()
                coord = torch.from_numpy(coord)
                # # #---->concat
                batch_feature = torch.cat((coord, features), dim=1)

                wsi_feature.append(batch_feature)

            #---->concat
            wsi_feature = torch.cat(wsi_feature)
            slide_name = Path(slide_id).name
            torch.save(wsi_feature, f'{save_dir}/{slide_name}.pt')