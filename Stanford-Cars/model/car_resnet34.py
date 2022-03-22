import torch
import torch.nn as nn
import torch.nn.functional as F

import resnet34 as res
import sys
sys.path.append('../../')
from MSHQP.config import resnet34_path

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

        self.sample_128 = Residual_Sampling(in_dim=128, out_dim=512, kernel_size=4, stride=4, padding=0)
        self.sample_256 = Residual_Sampling(in_dim=256, out_dim=512, kernel_size=2, stride=2, padding=0)

        self.fc_concat = torch.nn.Linear(512 ** 2 * 3, 196)
        self.fc_sum = torch.nn.Linear(512 ** 2, 196)

        self.softmax = nn.LogSoftmax(dim=1)
        self.features = res.resnet34(pretrained=True, model_root=resnet34_path)

        self.is_init = False
        self.feature_select_weight = nn.Parameter(torch.ones(15).cuda(), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(512)


    def extract_feature(self, outputs, batch_size):
        feature = []
        for i in range(1, len(outputs)):
            feature.append(self.bn(outputs[i]*outputs[0]).view(batch_size, 512, 14 ** 2))
        for i in range(1, 4):
            for j in range(4, 7):
                feature.append(self.bn(outputs[i]*outputs[j]).view(batch_size, 512, 14 ** 2))

        for i, fea in enumerate(feature):
            feature[i] = (torch.bmm(fea, torch.transpose(fea, 1, 2) / 14 ** 2)).view(batch_size, -1)

        if self.is_init is False:
            self.feature_select_weight.data = torch.mean(self.relu(torch.stack(feature)).view(15, -1), dim=1).data
            self.is_init = True

        self.feature_select_weight.data = self.relu(self.feature_select_weight.data)
        self.feature_select_weight.data = (self.feature_select_weight.data - torch.min(self.feature_select_weight.data)) \
                                          / (torch.max(self.feature_select_weight.data) - torch.min(
            self.feature_select_weight.data))

        feature_weight = self.feature_select_weight

        feature = torch.stack(feature) * feature_weight.view(15, 1, 1)

        select_feature = torch.cat([torch.nn.functional.normalize(
            torch.sign(feature[i, :, :]) * torch.sqrt(torch.abs(feature[i, :, :]) + 1e-10), dim=1)
                 for i in torch.topk(feature_weight, 3)[1]], dim=1)

        sum_feature = torch.mean(feature, 0)
        sum_feature = torch.nn.functional.normalize(torch.sign(sum_feature) * torch.sqrt(torch.abs(sum_feature) + 1e-10), dim=1)

        return select_feature.view(batch_size, -1), sum_feature


    def forward(self, x):
        batch_size = x.size(0)
        outputs = self.features(x)

        for index, fea in enumerate(outputs):
            if fea.size(1) == 128:
                outputs[index] = self.sample_128(fea)
            elif fea.size(1) == 256:
                outputs[index] = self.sample_256(fea)

        select_feature, sum_feature = self.extract_feature(outputs, batch_size)

        result = self.fc_concat(select_feature)
        sum_result = self.fc_sum(sum_feature)

        return self.softmax(result), self.softmax(sum_result), self.feature_select_weight