# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:25:33 2020

@author: hoon
"""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import os, sys, random
import numpy as np
import PIL
from PIL import Image

from gen_utils import *
from ds import *
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

load_tfm = transforms.Compose([
    transforms.ToTensor(),
    lambda x : (x-x.min())/(x.max()-x.min())
])

train_set = XrayDset('./data_new2/train/', load_tfm)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=10, shuffle=True)

test_set = XrayDset('./data4/test_Shenzen/', load_tfm)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=10, shuffle=False)

class XrayResnet(torch.nn.Module):
    def __init__(self):
        super(XrayResnet, self).__init__()
        self.C1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, stride=2)
        self.model_ft = torchvision.models.resnet18()
        self.model_ft.avgpool = torch.nn.AvgPool2d(kernel_size=4, padding=0, stride=2)
        self.model_ft.fc = torch.nn.Sequential(
            torch.nn.Linear(512,256),
            torch.nn.Linear(256,2)
        )
        
    def forward(self, x):
        y = x
        y = self.C1(y)
        for lid, layer in enumerate(list(self.model_ft.children())[:9]):
            y = layer(y)
        y = y.squeeze(-1).squeeze(-1)
        y = list(self.model_ft.children())[-1](y)
        return y
    
n_epochs = 30
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
M = XrayResnet()
M = M.to(device)
optimizer = torch.optim.Adam(M.parameters(), lr=6e-4, weight_decay=1e-2)
exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
criterion = torch.nn.CrossEntropyLoss()

train_loss_track = []
test_loss_track = []

for eph in range(n_epochs):
    print('epoch : {} ...'.format(eph))
    n_correct = 0
    avg_loss = 0
    n_samples = 0
    M.train()
    exp_lr_scheduler.step()
    for idx, xy in enumerate(train_loader):
        x, y = xy
        x, y = x.to(device), y.to(device)
        outputs = M(x)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        n_correct += torch.sum(preds.data == y.data)
        avg_loss += loss.item()
        n_samples += x.size(0)
    avg_loss = avg_loss/n_samples
    train_loss_track.append(avg_loss)
    print('train avg loss : ', avg_loss)
    print('num of correct samples : {}/{}'.format(n_correct, n_samples))
    
    n_correct = 0
    avg_loss = 0
    n_samples = 0
    gt_labels = []
    pred_labels = []
    M.eval()
    for idx, xy in enumerate(test_loader):
        x, y = xy
#        x, y = x.cuda(), y.cuda()
        x, y = x.to(device), y.to(device)
        outputs = M(x)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        
        n_correct += torch.sum(preds.data == y.data)
        gt_labels += list(y.data.cpu().numpy())
        pred_labels += list(preds.data.cpu().numpy())
        avg_loss += loss.item()
        n_samples += x.size(0)
    avg_loss = avg_loss/n_samples
    test_loss_track.append(avg_loss)
    print('test avg loss : ', avg_loss)
    print('num of correct samples : {}/{}'.format(n_correct, n_samples))
    
    
plt.plot(train_loss_track, 'b')
plt.plot(test_loss_track, 'r')
plt.xlabel('epochs')
plt.ylabel('avg loss')
plt.show()

target_names = ['No TB', 'TB', 'COVID']
print(classification_report(gt_labels, pred_labels, target_names=target_names))

