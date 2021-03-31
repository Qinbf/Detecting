# Detecting
 The platform for object detection research was implemented with **TensorFlow2** eager execution.
 
[![TensorFlow 2.4](https://img.shields.io/badge/TensorFlow-2.4-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.4.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

GitHub:[https://github.com/Qinbf/Detecting](https://github.com/Qinbf/Detecting)

B站关于项目的视频介绍：https://www.bilibili.com/video/BV1rZ4y1T7Nu

项目初衷是给大家提供一个既方便使用，同时又易于学习的目标检测工具。Detecting给大家提供多种预训练模型，可以直接下载使用，项目中的所有代码都有详细注释。

我先挖个坑，暂时只实现了[FasterRCNN](https://arxiv.org/abs/1506.01497)算法，后续会把坑填上，把主流的一些算法都实现。希望大家可以给个**Star**支持一下，谢谢！

如果有很多人喜欢Detecting这个项目的话，我会出一个免费的视频从头到尾讲解这个目标检测项目是如何做出来的。（一行一行代码讲，会讲到所有细节）

项目中肯定存在bug和不足，大家在使用时遇到问题或者有好的想法可以给我反馈。

 ------------------

 ## 安装
 首先确保已经安装Tensorflow2环境，然后再安装**detecting**模块。
 - **推荐使用pip安装：**
 ```sh
pip install detecting
```
- **也可以使用源码安装：**
  
先使用 `git` clone项目:
```sh
git clone https://github.com/Qinbf/detecting.git
```
 然后 `cd` 到detecting文件夹中执行安装命令:
```sh
cd detecting
sudo python setup.py install
```
- 如果需要训练或评估COCO数据集还需要安装[pycocotools](https://github.com/cocodataset/cocoapi)模块
------------------
## 快速使用
- **模型预测**

通常来说模型预测只需要几行代码
```python
from detecting.build.fasterrcnn import FasterRCNNModel
# 下载并载入预训练模型
# weights如果为'COCO'或'VOC'表示模型使用'COCO'或'VOC'数据集训练得到
# weights如果为'None'表示定义一个没有训练过的新模型
# weights如果为一个路径，表示从该路径载入训练好的模型参数
model = FasterRCNNModel(backbone='resnet101', weights='COCO', input_shape=(1024, 1024))
# 预测结果并显示
model.predict_show('test_images/000000018380.jpg')
```
<img  src="https://github.com/Qinbf/Detecting/blob/master/test_images/test.jpg" width="100%" height="100%">


------------------
## 训练新模型
使用Detecting训练自己的数据可以按照VOC数据集的格式先对数据进行标注。

标注工具:

- [Labelme](https://github.com/wkentaro/labelme)
  
- [LabelImg](https://github.com/tzutalin/labelImg)


下面把VOC数据集看成是我们自己标注好的新数据集。

- [VOC训练集下载](http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200803100327-c9y8o7303dsgsoco?attname=VOCtrainval_06-Nov-2007.tar&e=1596533588&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:eNBMJWzJvAXLUoMzn4sqBTyf60k=)

- [VOC测试集下载](http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200803100902-7egidf4rs2kggggg?attname=VOCtest_06-Nov-2007.tar&e=1596533617&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:TLRRBk3i0LerAb2XN4DDjnAs4cw=)

理论上训练集和测试集可以存放在任意位置，不过为了方便，大家可以参考我下面介绍的方式。我们可以新建一个datasets文件夹，然后把VOC训练集和测试集都放在datasets中，文件结构如下：

```
datasets/
└── VOC
    ├── test
    │   └── VOC2007
    │       ├── Annotations
    │       ├── ImageSets
    │       ├── JPEGImages
    │       ├── SegmentationClass
    │       └── SegmentationObject
    └── train
        └── VOC2007
            ├── Annotations
            ├── ImageSets
            ├── JPEGImages
            ├── SegmentationClass
            └── SegmentationObject
```
Annotations文件夹中保存这图片的标注，ImageSets文件夹保存这图片。我们可以自定义一个'train.yml'配置文件，配置文件内容如下：
```
DATASETS:
  NAMES: ('MYDATA')
  CLASSES: ('__background__', 
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
  IMAGE_DIR: ('datasets/VOC/train/VOC2007/JPEGImages')
  LABEL_DIR: ('datasets/VOC/train/VOC2007/Annotations')
  SCALE: (1024, 1024)

MODEL:
  BACKBONE: 'resnet101'
  WEIGHTS: 'COCO'
  INPUT_SHAPE: (1024, 1024)
  ANCHOR_SCALES: (64, 128, 256, 512)
  ANCHOR_FEATURE_STRIDES: (16, 16, 16, 16)
```
NAMES: ('MYDATA')表示训练自己的数据集

NAMES: ('COCO')表示训练'COCO'数据集

NAMES: ('VOC')表示训练'VOC'数据集

CLASSES: 设置数据集的标签

IMAGE_DIR: 设置图片路径

LABEL_DIR: 设置标注路径

SCALE: 生成器产生的图片尺寸

BACKBONE: 模型基本分类器

WEIGHTS: 模型权值。WEIGHTS如果为'COCO'或'VOC'表示模型使用'COCO'或'VOC'数据集训练得到
；WEIGHTS如果为'None'表示定义一个没有训练过的新模型；WEIGHTS如果为一个路径，表示从该路径载入训练好的模型参数

INPUT_SHAPE: 表示模型输入图片大小

ANCHOR_SCALES: anchors的大小

ANCHOR_FEATURE_STRIDES: anchors的步长

- **模型训练**

通常来说模型训练也只需要几行代码

```python
from detecting.build.fasterrcnn import FasterRCNNModel
from detecting.datasets.datasets import load_tf_dataset
from detecting.config import cfg
# 与配置文件中的配置合并
cfg.merge_from_file('train.yml')
# 载入数据集tf_dataset
tf_dataset = load_tf_dataset(cfg)
# 载入模型 
model = FasterRCNNModel(cfg)
# 训练模型
model.fit(tf_dataset)
```

本项目最重要的文件是detecting/config/defaults.py，里面保存着所有默认配置信息。我们可以自定义"*.yml"文件，用于修改默认配置信息。

更多使用方法可以查看[tutorial](https://github.com/Qinbf/Detecting/tree/master/tutorial)中的内容以及源代码。

------------------
 ## VOC 测试集实测结果
 | Detection Model | Backbone | Input resolution |  mAP |
 | :---- | :----: | :----: | :----: |
 | FasterRCNN| VGG16 | 1024x1024 | 53.97 |

  ------------------
 ## COCO 验证集实测结果
| Detection Model | Backbone | Input resolution |  AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| :---- | :----: | :----: | :----: | :----: |  :----: | :----: | :----: | :----: |
| FasterRCNN | ResNet50   |  640x640  |  24.7 | 39.9 | 26.0 | 5.7 | 26.1 | 42.6 |
| FasterRCNN | ResNet50   | 1024x1024 |  27.5 | 43.8 | 29.5 | 10.8 | 32.6 | 41.5 |
| FasterRCNN | ResNet101  | 640x640    | 27.0 | 41.2 | 29.2 | 7.2 | 28.6 | 45.0 |
| FasterRCNN | ResNet101  | 1024x1024  | 32.2 | 47.4 | 35.2 | 12.1 | 35.7 | 50.4 | 
| FasterRCNN | ResNet152  | 640x640    | 27.7 | 41.5 | 29.9 | 7.8 | 29.4 | 46.8 |
| FasterRCNN | ResNet152  | 1024x1024 | 32.0 | 46.7 | 35.2 | 11.4 | 35.3 | 51.6

  ------------------
 ## COCO数据集预训练模型下载地址  

[百度网盘](https://pan.baidu.com/s/1LXj-kj98FB2M-KACKfOUyw)  
密码: owi0

 ------------------
## Acknowledgment
[tensorflow/models/tree/master/research/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)

[Viredery/tf-eager-fasterrcnn](https://github.com/Viredery/tf-eager-fasterrcnn)

[matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
