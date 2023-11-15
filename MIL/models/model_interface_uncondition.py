import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd

#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import cross_entropy_torch

#---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

#---->
import pytorch_lightning as pl


#---->diffusion
import sys
sys.path.append("/data111/shaozc/Diffusion/latent-diffusion-main/") ## add the dir of latent diffusion
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler




#--------------->load pretrained weight
def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # model.cuda()
    model.eval()
    return model

#---->加载模型和采样器 基准测试的时候注释！！！！！！
#---->Unitopatho
config = OmegaConf.load("/data111/shaozc/Diffusion/latent-diffusion-main/configs/latent-diffusion/unitopatho_uncondition.yaml")  
aug_model_pre = load_model_from_config(config, "/data111/shaozc/Diffusion/latent-diffusion-main/logs/2023-02-20T21-19-00_unitopatho_uncondition/checkpoints/epoch=000368.ckpt") 
#---->uncondition
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
aug_model = aug_model_pre.to(device)
aug_model.eval()
sampler = DDIMSampler(aug_model)

class ModelInterfaceUncondition(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterfaceUncondition, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']

        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(num_classes = self.n_classes, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = self.n_classes,
                                                                           average='micro'),
                                                     torchmetrics.CohenKappa(num_classes = self.n_classes),
                                                     torchmetrics.F1(num_classes = self.n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = self.n_classes),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = self.n_classes),
                                                     torchmetrics.Specificity(average = 'macro',
                                                                            num_classes = self.n_classes)])
        else : 
            self.AUROC = torchmetrics.AUROC(num_classes=2, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = 2,
                                                                           average = 'micro'),
                                                     torchmetrics.CohenKappa(num_classes = 2),
                                                     torchmetrics.F1(num_classes = 2,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = 2)])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        #---->diffusion 基准测试的时候注释！！！！！！
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0
        self.total_steps = kargs['data'].total_steps
        self.u_scale = kargs['data'].u_scale
        self.scale = kargs['data'].scale


    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        #---->inference
        data, label = batch

        #---->diffusion 基准测试的时候注释！！！！！！
        with torch.no_grad():
            WSI_pt = data[..., 2:].squeeze(0) #batch size=1
            position_pt = data[..., :2]
            n_samples_per_class = len(WSI_pt)


            #---->ablation！！！！！！
            samples_ddim, _ = sampler.sample(S=self.total_steps,
                                            batch_size=n_samples_per_class,
                                            shape=[WSI_pt.shape[-1], 1],
                                            verbose=False,
                                            nu = self.scale,
                                            x_T=WSI_pt, #输入原始特征
                                            eta=0.0)
            x_samples_ddim = aug_model.decode_first_stage(samples_ddim)
            features = x_samples_ddim.permute(2, 0, 1) #[1, n, 512]
            data = torch.cat((position_pt, features), dim=-1)




        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        loss = self.loss(logits, label)


        return {'loss': loss} 

    def validation_step(self, batch, batch_idx):
        #---->inference
        data, label = batch

        #---->diffusion 基准测试的时候注释！！！！！！
        with torch.no_grad():
            WSI_pt = data[..., 2:].squeeze(0) #batch size=1
            position_pt = data[..., :2]
            n_samples_per_class = len(WSI_pt)



            #---->ablation！！！！！！
            samples_ddim, _ = sampler.sample(S=self.total_steps,
                                            batch_size=n_samples_per_class,
                                            shape=[WSI_pt.shape[-1], 1],
                                            verbose=False,
                                            nu = self.scale,
                                            x_T=WSI_pt, #输入原始特征
                                            eta=0.0)
            x_samples_ddim = aug_model.decode_first_stage(samples_ddim)
            features = x_samples_ddim.permute(2, 0, 1) #[1, n, 512]
            data = torch.cat((position_pt, features), dim=-1)

        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']


        #---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}


    def validation_epoch_end(self, val_step_outputs):
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.stack([x['label'] for x in val_step_outputs], dim = 0)
        
        #---->
        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)

        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        

    


    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def test_step(self, batch, batch_idx):
        #---->inference
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def test_epoch_end(self, output_results):
        probs = torch.cat([x['Y_prob'] for x in output_results], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        target = torch.stack([x['label'] for x in output_results], dim = 0)
        
        #---->
        auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
        metrics['auc'] = auc
        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()
        print()
        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        #---->
        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path / 'result.csv')


    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)