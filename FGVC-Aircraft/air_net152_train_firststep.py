# encoding:utf-8
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

import data.air_data as data
import model.air_resnet152 as MSHQP
from utils.utils import progress_bar
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

trainset = data.MyDataset('data/air_trainlist_shuffle.txt', transform=transforms.Compose([
                            transforms.Resize((500, 480), Image.BILINEAR),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(448),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                            ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8)
testset = data.MyDataset('data/air_testlist.txt', transform=transforms.Compose([
                            transforms.Resize((500, 480), Image.BILINEAR),
                            transforms.CenterCrop(448),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                            ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=8)
cudnn.benchmark = True

model = MSHQP.Net()
model.cuda()

criterion = nn.NLLLoss()
lr = 1.0
model.features.requires_grad = False
optimizer = optim.SGD([
                        {'params': model.sample_512.parameters(), 'lr': lr},
                        {'params': model.sample_1024.parameters(), 'lr': lr},
                        {'params': model.sample_2048.parameters(), 'lr': lr},
                        {'params': model.feature_select_weight, 'lr': 1e-2},
                        {'params': model.bn.parameters(), 'lr': lr},
                        {'params': model.fc_concat.parameters(), 'lr': lr},
                        {'params': model.fc_sum.parameters(), 'lr': lr},
], lr=1, momentum=0.9, weight_decay=1e-5)


def train(epoch):
    model.train()
    print('----------------------------------------Epoch: {}----------------------------------------'.format(epoch))
    for batch_idx, (img, target) in enumerate(trainloader):
        img, target = img.cuda(), target.cuda()
        model.zero_grad()
        output, sum_output, feature_weight = model(img)
        sparse_loss = 1e-2 * torch.norm(feature_weight, p=1, dim=0)
        supervisory_loss = 1e-2 * criterion(sum_output, target)
        cls_loss = criterion(output, target)
        loss = sparse_loss + cls_loss + supervisory_loss
        loss.backward()
        optimizer.step()
        progress_bar(batch_idx, len(trainloader), 'total_loss: ' + str('{:.4f}'.format(loss.data.item()))
                     + ' cls_loss: ' + str('{:.4f}'.format(cls_loss.data.item()))
                     + ' supervisory_loss: ' + str('{:.4f}'.format(supervisory_loss.data.item()))
                     + ' sparse_loss: ' + str('{:.4f}'.format(sparse_loss.data.item())) + ' | train')


def test():
    model.eval()
    print('----------------------------------------Test---------------------------------------------')
    test_loss = 0
    correct = 0
    sum_correct = 0
    with torch.no_grad():
        for batch_idx, (img, target) in enumerate(testloader):
            img, target = img.cuda(), target.cuda()
            output, sum_output, feature_weight = model(img)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            sum_pred = sum_output.data.max(1, keepdim=True)[1]
            sum_correct += sum_pred.eq(target.data.view_as(sum_pred)).cpu().sum()
            progress_bar(batch_idx, len(testloader), 'test')

    test_loss /= len(testloader)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss * 8, correct, len(testloader.dataset),
        100.0 * float(correct) / len(testloader.dataset)))


def adjust_learning_rate(optimizer, epoch):
    if epoch % 40 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


for epoch in range(1, 81):
    train(epoch)
    if epoch % 5 == 0:
        test()
    adjust_learning_rate(optimizer, epoch)


torch.save(model.state_dict(), 'firststep_air_net152.pth')
