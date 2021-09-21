# HW5 CNN

**说明：**

在`code`文件夹里包含了本次实验代码。

## LeNet5对MNIST训练

`LeNet5.py`是用LeNet5神经网络对MNIST进行训练与测试。

用法：

```
python LeNet5.py <train/test> <model_path>
```

已经将训练model保存为`LeNet5.pth`

运行后会自己下载MNIST数据集

实验的结果保存在`LeNet.log`

P.S. `LeNetVis.py`是结果的可视化

## AlexNet对CIFAR-10训练

`AlexNet.py`是用AlexNet神经网络对CIFAR-10进行训练与测试。

用法：

```
python AlexNet.py <train/test> <model_path>
```

因为数据集和训练的模型都实在是太大了（超过100M），所以就不上传了，如有需要可以在百度网盘下载（数据集也在这里）：

```
链接：https://pan.baidu.com/s/1DNPmH-d62njXJeCZEucbGw 
提取码：cv05 
复制这段内容后打开百度网盘手机App，操作更方便哦--来自百度网盘超级会员V3的分享
```



实验的结果保存在`AlexNet.log`



**更多请参考实验报告**