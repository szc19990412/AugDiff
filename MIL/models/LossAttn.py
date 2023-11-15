import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#---->https://github.com/xsshi2015/Loss-Attention/blob/master/MIL/model.py

#----> Attention module
class AttentionLayer(nn.Module):
    def __init__(self, dim=512):
        super(AttentionLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_1, b_1, flag):
        if flag==1:
            out_c = F.linear(features, W_1, b_1)
            out = out_c - out_c.max()
            out = out.exp()
            out = out.sum(1, keepdim=True)
            alpha = out / out.sum(0)
            

            alpha01 = features.size(0)*alpha.expand_as(features)
            context = torch.mul(features, alpha01)
        else:
            context = features
            alpha = torch.zeros(features.size(0),1)
                
        return context, out_c, torch.squeeze(alpha)

class LossAttn(nn.Module):
    def __init__(self, n_classes, feats_size=512):
        super(LossAttn, self).__init__()
        self.fc = nn.Sequential(nn.Linear(feats_size, 512), nn.ReLU()) #512->512
        self.attention_net = AttentionLayer(512)
        
        self.classifiers = nn.Linear(512, n_classes)

    def forward(self, **kwargs):

        h = kwargs['data'].float()[..., 2:] #[B, n, 512]
        h = h.squeeze() #[n, 512]
        h = self.fc(h)
        
        #---->Attention
        out, out_c, alpha = self.attention_net(h, self.classifiers.weight, self.classifiers.bias, 1) 
        out = out.mean(0,keepdim=True)

        #---->predict output
        logits = self.classifiers(out) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

