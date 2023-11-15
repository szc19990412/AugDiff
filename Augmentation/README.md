### Thanks for the open source at https://github.com/gatsby2016/Augmentation-PyTorch-Transforms

SICAP v2: https://data.mendeley.com/datasets/9xxm58dvs3/1
UnitoPatho: https://github.com/EIDOSLAB/UNITOPATHO    split [unitopath-public/800/*] as guided in Augmentation/unitopatho_split.txt
TMA: https://github.com/uw-loci/gnn-pccp


### (1) Doing image augmentation first
```python
python dataAug_myTransforms.py
```
### (2) Extracting features for MIL training, one WSI conresponds to one pt file
```python
python extract_feature_wsi.py
```
### (3) Extracting features for diffusion training, one patch conresponds to one pt file. This way can help diffusion training have larger batch size. 
```python
python extract_feature.py
```