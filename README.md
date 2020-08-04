# Detecting
 The platform for object detection research was implemented with **TensorFlow2** eager execution.
 
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

GitHub:[https://github.com/Qinbf/Detecting](https://github.com/Qinbf/Detecting)

项目初衷是给大家提供一个即方便使用，同时又易于学习的目标检测工具。Detecting给大家提供多种预训练模型，可以直接下载使用，项目中的所有代码都有详细注释。

我先挖个坑，暂时只实现了[FasterRCNN](https://arxiv.org/abs/1504.08083)算法，后续会把坑填上，把主流的一些算法都实现。希望大家可以给个**Star**支持一下，谢谢！

如果有很多人喜欢Detecting这个项目的话，我会出一个免费的视频从头到尾讲解这个目标检测项目是如何做出来的。（一行一行代码讲，会讲到所有细节）

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
from detecting.build.fasterrcnn import FasterRCNNModelplot_model
# 下载并载入预训练模型
model = FasterRCNNModel(backbone='resnet101', weights='COCO', input_shape=(1024, 1024))
# 预测结果并显示
model.predict_show('test_images/000000018380.jpg')
```
<img  src="https://raw.githubusercontent.com/Qinbf/detecting/master/test_images/test_images.jpg" width="50%" height="50%">

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

- **更多使用方法可以查看tutorial中的内容以及源代码**

------------------
 ## VOC 测试集实测结果
 | Detection Model | Backbone | Input resolution |  mAP |
 | :---- | :----: | :----: | :----: |
 | [FasterRCNN](http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200730101828-39q2dxxb292ccwk4?attname=fasterrcnn_vgg16_1024_VOC.zip&e=1596190406&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:2H1LPFT0GEIwqU-sBqg4yGNWXm4=) | VGG16 | 1024x1024 | 53.97 |

  ------------------
 ## COCO 验证集实测结果
| Detection Model | Backbone | Input resolution |  AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| :---- | :----: | :----: | :----: | :----: |  :----: | :----: | :----: | :----: |
| [FasterRCNN](http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200727093036-qyhky8vjois40s8o?attname=fasterrcnn_resnet50_640.zip&e=1595860441&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:y_jGmVl-uexuoKYbQJkcvOAJkK4=) | ResNet50   |  640x640  |  24.7 | 39.9 | 26.0 | 5.7 | 26.1 | 42.6 |
| [FasterRCNN](http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200728094544-j9vusnec9uokk8sw?attname=fasterrcnn_resnet50_1024.zip&e=1595904466&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:8aMMMnTS3gsWr_d3KbRS07Hc2DM=) | ResNet50   | 1024x1024 |  27.5 | 43.8 | 29.5 | 10.8 | 32.6 | 41.5 |
| [FasterRCNN](http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200728095345-mild1p6duzkg80ww?attname=fasterrcnn_resnet101_640.zip&e=1595905768&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:JMGAxZR9RAiNz4CRc1bld2UgX3E=) | ResNet101  | 640x640    | 27.0 | 41.2 | 29.2 | 7.2 | 28.6 | 45.0 |
| [FasterRCNN](http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200724041434-5aopoxzxgc4coc80?attname=fasterrcnn_resnet101_1024.zip&e=1595582245&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:8AKzHeuM0vBUfWplGOPzHTVe4BQ=) | ResNet101  | 1024x1024  | 32.2 | 47.4 | 35.2 | 12.1 | 35.7 | 50.4 | 
| [FasterRCNN](http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200728105534-ijwagx1ofxw8044w?attname=fasterrcnn_resnet152_640.zip&e=1595908752&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:oo0SpAvx3496ZqPMCDG6UkHlQ7k=) | ResNet152  | 640x640    | 27.7 | 41.5 | 29.9 | 7.8 | 29.4 | 46.8 |
| [FasterRCNN](http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200728110123-96bcwki8qfoc4gco?attname=fasterrcnn_resnet152_1024.zip&e=1595909196&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:rEiMY--bK2VkoPFAOn7Kgc5pLdE=) | ResNet152  | 1024x1024 | 32.0 | 46.7 | 35.2 | 11.4 | 35.3 | 51.6


 ------------------
## Acknowledgment:
[tensorflow/models/tree/master/research/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)

[Viredery/tf-eager-fasterrcnn](https://github.com/Viredery/tf-eager-fasterrcnn)

[matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
