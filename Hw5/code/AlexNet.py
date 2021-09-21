import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import transforms
import sys

BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001
momentum = 0.9

# 显示部分图片
def imshow(img):
    img = img/2 + 0.5
    nping = img.numpy()
    plt.imshow(np.transpose(nping, (1,2,0)))
    plt.show()
 
# 图像预处理与加载
transform = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data',train=True, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data',train=False, download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

classes_names = trainset.classes

# AlexNet网络
class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3,96,11,4)                       # 卷积层1，卷积核个数96，大小为11*11*3，步长为4
        self.conv2 = nn.Conv2d(96,256,5,padding=2,groups=2)     # 卷积层2，卷积核个数256，大小为5*5*48，步长为2，填充为2
        self.conv3 = nn.Conv2d(256,384,3,padding=1)             # 卷积层3，卷积核个数384，大小为3*3*256，填充为1
        self.conv4 = nn.Conv2d(384,384,3,padding=1, groups=2)   # 卷积层4，卷积核个数384，大小为3*3，填充为1
        self.conv5 = nn.Conv2d(384,256,3,padding=1, groups=2)   # 卷积层5，卷积核个数256，大小为3*3，填充为1
        self.fc1 = nn.Linear(256*6*6,4096)                      # 全连接层1，神经元个数4096
        self.fc2 = nn.Linear(4096,4096)                         # 全连接层2，神经元个数4096
        self.fc3 = nn.Linear(4096,num_classes)                  # 全连接层3，神经元个数4096
 
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))           # 对卷积层1的池化
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))           # 对卷积层2的池化
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)),(2,2))           # 对卷积层5的池化
        x = x.view(x.size(0),256*6*6)
        x = F.dropout(F.relu(self.fc1(x)),p=0.5)                # 全连接层使用dropout和relu
        x = F.dropout(F.relu(self.fc2(x)),p=0.5)
        x = self.fc3(x)
        return x


def train(model, device, train_loader, optimizer, criterion, epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, lables = data
        inputs = inputs.to(device)
        lables = lables.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, lables)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d,%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
            
    
def test(model, device, train_loader, test_loader):
    # 检测训练图片中的正确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 50000 train images: %d %%'%(100*correct/total))
    
    # 检测测试图片中的正确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%'%(100*correct/total))
    
    # 检测每个标签的正确率
    class_correct = [0.0]*10
    class_total = [0.0]*10
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data,1)
            c = (predicted == labels)
            if len(c) == 16:
                for i in range(16):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += c[i].item()
            else:
                for i in range(32):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
    
    
    for i in range(10):
        print('Accuracy of %5s : %2d %%'%(classes_names[i], 100*class_correct[i]/class_total[i]))



if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('usage: python AlexNet.py <train/test> <model_path>')
        sys.exit(1)
    

    if sys.argv[1] == 'train':
        # training 
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        imshow(torchvision.utils.make_grid(images))
        print(' '.join('%5s'%classes_names[labels[j]] for j in range(BATCH_SIZE)))

        model = AlexNet()
        model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        for epoch in range(0, EPOCHS):
            train(model, DEVICE, train_loader, optimizer, criterion, epoch)
            
        print('Finished Training')
        torch.save(model, sys.argv[2])
    

    elif sys.argv[1] == 'test':
        # testing
        model = torch.load(sys.argv[2])
        print(model)
        
        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s'%classes_names[labels[j]] for j in range(32)))
        
        
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        predicted = torch.argmax(outputs,1)
        print('Predicted: ',' '.join('%5s'%classes_names[predicted[j]] for j in range(32)))
        
        test(model, DEVICE, train_loader, test_loader)
    
    
    