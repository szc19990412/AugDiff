
### Basic MIL Training
```python
for((FOLD=0;FOLD<4;FOLD++));
do
    CUDA_VISIBLE_DEVICES=0 python train_unito_base.py --stage='train' --config='UnitoPatho/AMIL.yaml'  --gpus=0 --fold=$FOLD
    CUDA_VISIBLE_DEVICES=0 python train_unito_base.py --stage='test' --config='UnitoPatho/AMIL.yaml'  --gpus=0 --fold=$FOLD
done
python metric.py --config='UnitoPatho/AMIL.yaml' 
```


### MIL Training with Image Augmentation
```python
for((FOLD=0;FOLD<4;FOLD++));
do
    CUDA_VISIBLE_DEVICES=0 python train_unito_base.py --stage='train' --config='UnitoPatho/AMIL_aug_image.yaml'  --gpus=0 --fold=$FOLD
    CUDA_VISIBLE_DEVICES=0 python train_unito_base.py --stage='test' --config='UnitoPatho/AMIL_aug_image.yaml'  --gpus=0 --fold=$FOLD
done
python metric.py --config='UnitoPatho/AMIL_aug_image.yaml' 
```


### MIL Training with Feature Augmentation
```python
for((FOLD=0;FOLD<4;FOLD++));
do
    CUDA_VISIBLE_DEVICES=0 python train_unito.py --stage='train' --config='UnitoPatho/AMIL_aug.yaml'  --gpus=0 --fold=$FOLD
    CUDA_VISIBLE_DEVICES=0 python train_unito.py --stage='test' --config='UnitoPatho/AMIL_aug.yaml'  --gpus=0 --fold=$FOLD
done
python metric.py --config='UnitoPatho/AMIL_aug.yaml' 
```