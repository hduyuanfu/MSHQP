import torch
import torch.nn as nn
import torch.nn.functional as F

import resnet152 as res
import sys
sys.path.append('../../')
from MSHQP.config import resnet152_path

class lower_channels(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(lower_channels, self).__init__()
        
        self.main__ = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_dim),
        )
        

    def forward(self, x):
        x = self.main__(x)
        return x


class Residual_Sampling(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding):
        super(Residual_Sampling, self).__init__()
        self.Maxpool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride=stride, padding=padding),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.res = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False),
        )
        self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.Maxpool1(x) + self.res(x)
        x = self.bn(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.bn = nn.BatchNorm2d(512)

        self.sample_512 = Residual_Sampling(in_dim=512, out_dim=512, kernel_size=4, stride=4, padding=0)
        self.sample_1024 = Residual_Sampling(in_dim=1024, out_dim=512, kernel_size=2, stride=2, padding=0)
        self.sample_2048 = lower_channels(in_dim=2048, out_dim=512)
 
        self.fc_concat = torch.nn.Linear(512 ** 2 * 3, 196)
        self.fc_sum = torch.nn.Linear(512 ** 2, 196)
      
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.features = res.resnet152(pretrained=True, model_root=resnet152_path)
      
        self.is_init = False
        self.feature_select_weight = nn.Parameter(torch.ones(15).cuda(), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)
        
     
    def extract_feature(self, outputs, batch_size):
        feature = []
        for i in range(1, len(outputs)):
            feature.append(self.bn(outputs[i] * outputs[0]).view(batch_size, 512, 14 ** 2))
        for i in range(1, 4):
            for j in range(4, 7):
                feature.append(self.bn(outputs[i] * outputs[j]).view(batch_size, 512, 14 ** 2))

        for i, fea in enumerate(feature):
            feature[i] = (torch.bmm(fea, torch.transpose(fea, 1, 2) / 14 ** 2)).view(batch_size, -1)

        if self.is_init is False:
            self.feature_select_weight.data = torch.mean(self.relu(torch.stack(feature)).view(15, -1), dim=1).data
            self.is_init = True

        self.feature_select_weight.data = self.relu(self.feature_select_weight.data)
        self.feature_select_weight.data = (self.feature_select_weight.data - torch.min(self.feature_select_weight.data)) \
                                          / (torch.max(self.feature_select_weight.data) - torch.min(self.feature_select_weight.data))

        feature_weight = self.feature_select_weight
        feature = torch.stack(feature) * feature_weight.view(15, 1, 1)

        select_feature = torch.cat([torch.nn.functional.normalize(
            torch.sign(feature[i, :, :]) * torch.sqrt(torch.abs(feature[i, :, :]) + 1e-10), dim=1)
                 for i in torch.topk(feature_weight, 3)[1]], dim=1)

        sum_feature = torch.mean(feature, 0)
        sum_feature = torch.nn.functional.normalize(torch.sign(sum_feature) * torch.sqrt(torch.abs(sum_feature) + 1e-10), dim=1)

        return select_feature.view(batch_size, -1), sum_feature


    def hook_fn_forward(self, module, input, output):
        self.total_fea_out.append(output)


    def forward(self, x):
        batch_size = x.size(0)

        self.total_fea_out = []
        self.hook_list = []
        target_layers = ['layer2.7', 'layer3.11', 'layer3.23', 'layer3.35', 'layer4.0', 'layer4.1', 'layer4.2']
        modules = self.features.named_modules()
        for name, module in modules:
            if name in target_layers:
                handle = module.register_forward_hook(self.hook_fn_forward)  
                self.hook_list.append(handle)
        self.features(x)
        for handle in self.hook_list:
            handle.remove() 

        for index, fea in enumerate(self.total_fea_out):
            if fea.size(1) == 512:
                self.total_fea_out[index] = self.sample_512(fea)
            elif fea.size(1) == 1024:
                self.total_fea_out[index] = self.sample_1024(fea)
            elif fea.size(1) == 2048:
                self.total_fea_out[index] = self.sample_2048(fea)

        select_feature, sum_feature = self.extract_feature(self.total_fea_out, batch_size)

        result = self.fc_concat(select_feature)
        sum_result = self.fc_sum(sum_feature)
        return self.logsoftmax(result), self.logsoftmax(sum_result), self.feature_select_weight
