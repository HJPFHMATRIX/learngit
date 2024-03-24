import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import torchvision






dateset = torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),  # 将数据转换为tensor格式,下载的数据是numpy格式
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   # 正则化数据，神经网络的数据最好在0附近均匀分布，但是图片的参数是在0到1之间，所以需要正则化数据(提高性能)
                               ])),  # 下载数据集,用于训练


dateset_2 = torchvision.datasets.CIFAR10('cifar10_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),  # 将数据转换为tensor格式,下载的数据是numpy格式
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   # 正则化数据，神经网络的数据最好在0附近均匀分布，但是图片的参数是在0到1之间，所以需要正则化数据(提高性能)
                               ])),  # 下载数据集,用于测试

dataloader = DataLoader(dateset, batch_size=64, shuffle=True, num_workers=2, drop_last=False)
dataloader_2 = DataLoader(dateset_2, batch_size=64, shuffle=True, num_workers=2, drop_last=False)



print(dateset)
print(dataloader)
print(dateset_2)