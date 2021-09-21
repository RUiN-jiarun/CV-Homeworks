# Eigenface 
## 数据集

需要下载`att_faces`文件夹，放到`code`里

```
链接：https://pan.baidu.com/s/1UnKXiQ_TeF7jzDL-WvbPuA 
提取码：cv99 
--来自百度网盘超级会员V4的分享
```

`att_faces`文件夹里包含41个文件夹，命名为`s1`~`s41`，每个文件夹包含同一个人的10张人脸照片。  
其中前40个文件夹为AT&T数据集的内容，`s41`文件夹里是我自己的脸。  
每张照片的大小为112x92，格式为pgm格式。

## 训练过程
```cmd
>> python mytrain.py <att_faces_dir> <export_model_path> [<energy>]
```
例如：
```cmd
>> python mytrain.py att_faces mymodel.json 0.75
```
注意：导出模型需要花很长的时间！！！所以如果助教老师要检查代码时请注释掉mytrain.py中的这一句：
```python
export_model(mean_face, eigen_face, diffTrain, str(sys.argv[2]))
```
但是命令行操作时依然要把导出模型路径写上。

以下链接可以下载`model_full.json`

```
链接：https://pan.baidu.com/s/1ClJlRAYfrQKOt3GVRNvXPA 
提取码：cv98 
--来自百度网盘超级会员V4的分享
```



## 检测过程
```cmd
>> python mytest.py <test_iamge_dir> <import_model_path>
```
例如：
```cmd
>> python mytest.py att_faces/s1/6.pgm model.json
```
注意：已经有两个训练好的模型`model.json`和`model_full.json`，可以直接导入。
## 重建过程
```cmd
>> python myreconstruct.py <reconstruct_image_dir> <import_model_path>
```
例如：
```cmd
>> python myreconstruct.py att_faces/s41/10.pgm model.json
```
## 其他文件说明
* `myface.py`：进行自己人脸数据处理与构建的脚本
* `rank-1.py`：分析准确率随PCs变化的脚本
* `haarcascade_frontalface_default.xml`：opencv人脸识别的模型