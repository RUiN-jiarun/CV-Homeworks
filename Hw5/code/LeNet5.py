# python3
# coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import sys


BATCH_SIZE = 512        # 批的大小
EPOCHS = 10             # 遍历训练集的次数
DEVICE = torch.device('cuda')   # 分配设备
lr = 0.001              # 学习率
momentum = 0.9

#  定义网络
class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()      # 继承父类nn.Model的属性并初始化
        self.conv1 = nn.Conv2d(1, 6, 5)     # 卷积层1——LeNet5第一层
        self.conv2 = nn.Conv2d(6, 16, 5)    # 卷积层2——LeNet5第三层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # 全连接层1——LeNet5第五层
        self.fc2 = nn.Linear(120, 84)       # 全连接层2——LeNet5第六层
        self.fc3 = nn.Linear(84, 10)        # 全连接层output——LeNet第七层

    def forward(self, x):
        """正向传导过程

        :param x: 输入样本
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) # 池化层1——LeNet5第二层
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2)) # 池化层2——LeNet5第四层
        # x = x.view(-1, self.num_flat_features(x))   # 数据重组为256*(16*5*5)
        x = x.view(-1, reduce(lambda x,y:x*y, x.size()[1:]))   # 数据重组为256*(16*5*5)
        x = F.relu(self.fc1(x))             # 激活函数，产生新的输出
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 用torch的图像处理库预处理
transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    # # this line for a glimpse of the data
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# 下载MNIST数据库，设定数据加载器
train_set = datasets.MNIST(root='data', train=True, download=True, transform=transforms)
test_set = datasets.MNIST(root='data', train=False, download=True, transform=transforms)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


def train(model, device, train_loader, optimizer, criterion, epoch):
    """训练过程

    :param model: 输入网络模型
    :param device: 分配设备
    :param train_loader: 训练数据集加载
    :param optimizer: 优化函数
    :param criterion: 损失函数
    :param epoch: 训练总次数
    """
    model.train()   # 启用 BatchNormalization 和 Dropout，让model变成训练模式，起到防止网络过拟合的问题
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion):
    """测试过程

    :param model: 输入网络模型
    :param device: 分配设备
    :param test_loader: 测试数据集加载
    :param criterion: 损失函数
    """
    model.eval()    # 不启用 BatchNormalization 和 Dropout，不会取平均，而是用训练好的值
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # # 显示一个batch的数据
    # images, labels = next(iter(trainloader))
    # img = torchvision.utils.make_grid(images)
    # img = img.numpy().transpose(1, 2, 0)
    # std = [0.1304,]
    # mean = [0.3081,]
    # img = img * std + mean
    # plt.imshow(img)
    # plt.show()

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('usage: python LeNet5.py <train/test> <model_path>')
        sys.exit(1)

    if sys.argv[1] == 'train':
        # 训练过程
        model = LeNet5().to(DEVICE)                         # 构建训练网络
        optimizer = optim.Adam(model.parameters(), lr=lr)   # 优化算法：Adam算法
        criterion = nn.CrossEntropyLoss(reduction='sum')    # 损失算法：交叉熵函数
        for epoch in range(1, EPOCHS + 1):
            train(model, DEVICE, train_loader, optimizer, criterion, epoch)
            test(model, DEVICE, test_loader, criterion)
        torch.save(model, sys.argv[2])         # 保存模型
        print('trianing done.')

    elif sys.argv[1] == 'test':
        # 加载模型并测试
        model = torch.load(sys.argv[2])
        criterion = nn.CrossEntropyLoss(reduction='sum')
        test(model, DEVICE, test_loader, criterion)
