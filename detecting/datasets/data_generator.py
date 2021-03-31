import numpy as np
from tensorflow.keras.utils import Sequence
from detecting.datasets import transforms,utils
import os
import cv2

class DataGenerator(Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataset))
        # 打乱索引
        if self.shuffle:
            np.random.shuffle(self.indices)

    # 获得index位置的批次数据
    def __getitem__(self, index):
        img_list = []
        img_meta_list = []
        bbox_list = []
        label_list = []
        # 循环一个批次的数据
        for i in self.indices[index * self.batch_size: (index + 1) * self.batch_size]:
            # 获取数据
            img, img_meta, bbox, label = self.dataset[i]
            img_list.append(img)
            img_meta_list.append(img_meta)
            bbox_list.append(bbox)
            label_list.append(label)

        if self.batch_size == 1:
            batch_imgs = np.expand_dims(img_list[0], 0)
            batch_metas = np.expand_dims(img_meta_list[0], 0)
            batch_bboxes = np.expand_dims(bbox_list[0], 0)
            batch_labels = np.expand_dims(label_list[0], 0)
        else:
            # 堆叠为一个批次的数据
            batch_imgs = np.stack(img_list,axis=0)
            batch_metas = np.stack(img_meta_list,axis=0)
            # 先填充
            bbox_list = self._pad_batch_data(bbox_list)
            batch_bboxes = np.stack(bbox_list,axis=0)
            # 先填充
            label_list = self._pad_batch_data(label_list)
            batch_labels = np.stack(label_list,axis=0)
        # 返回
        return batch_imgs, batch_metas, batch_bboxes, batch_labels

    def __call__(self):
        for img_idx in self.indices:
            img, img_meta, bbox, label = self.dataset[img_idx]
            yield img, img_meta, bbox, label

    # 返回生成器的长度
    def __len__(self):
        return int(np.ceil(float(len(self.dataset))/self.batch_size))  

    # 用于填充bbox和label，使得一个批次中的bbox和label的数量相等
    def _pad_batch_data(self, data):
        # 计算这个批次中最多的数据量
        max_len = max([d.shape[0] for d in data])
        temp_list = []
        for d in data:
            # 计算需要填充的数据量
            pad_len = max_len - d.shape[0]
            if d.ndim == 1:
                # 填充pad_len个数据0
                d = np.pad(d,(0,pad_len),'constant',constant_values=0)
            elif d.ndim == 2:
                # 填充pad_len行数据0
                d = np.pad(d,((0,pad_len),(0,0)),'constant',constant_values=0)
            temp_list.append(d)
        return temp_list

    # 在epoch末尾
    def on_epoch_end(self):
        # 打乱数据索引
        if self.shuffle: 
            np.random.shuffle(self.indices)
    
    # 获得种类名称
    def get_categories(self):
        return self.dataset.get_categories()

    # 类别数
    def num_classes(self):
        return len(self.get_categories())
    
    # 返回数据集大小
    def size(self):
        return len(self.dataset) 

# 测试数据生成器
class TestDataGenerator(Sequence):
    def __init__(self, cfg, batch_size=1, shuffle=False):
        self.dataset_dir = cfg.DATASETS.IMAGE_DIR
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_transform = transforms.ImageTransform(cfg.DATASETS.SCALE,
                                                        cfg.DATASETS.IMG_MEAN,
                                                        cfg.DATASETS.IMG_STD,
                                                        cfg.DATASETS.PAD_MODE,
                                                        cfg.DATASETS.KEEP_ASPECT)
        # 读取所有图片
        self.img_list = self._get_all_images(self.dataset_dir)
        self.indices = np.arange(len(self.img_list))
        # 打乱索引
        if self.shuffle:
            np.random.shuffle(self.indices)

    # 获得index位置的批次数据
    def __getitem__(self, index):
        batch_img = []
        batch_img_meta = []
        # 循环一个批次的数据
        for i in self.indices[index * self.batch_size: (index + 1) * self.batch_size]:
            # 读取图片
            imgs = cv2.imread(self.img_list[i], cv2.IMREAD_COLOR)
            # BGR转RGB
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
            # 原始图片shape
            ori_shape = imgs.shape
            # 图片不翻转
            flip = False
            # 得到填充后的图片，resize后图片shape，缩放因子
            imgs, img_shape, scale_factor = self.img_transform(imgs, flip)
            # 填充后的图片shape
            pad_shape = imgs.shape
            img_meta_dict = dict({
            'ori_shape': ori_shape,
            'img_shape': img_shape,
            'pad_shape': pad_shape,
            'scale_factor': scale_factor,
            'flip': flip
            })
            # 把img_meta_dict中的数据组成1维的array
            img_meta = utils.compose_image_meta(img_meta_dict)

            batch_img.append(imgs)
            batch_img_meta.append(img_meta)

        if self.batch_size == 1:
            batch_imgs = np.expand_dims(batch_img[0], 0)
            batch_metas = np.expand_dims(batch_img_meta[0], 0)
        else:
            # 堆叠为一个批次的数据
            batch_imgs = np.stack(batch_img,axis=0)
            batch_metas = np.stack(batch_img_meta,axis=0)
        # 返回
        return batch_imgs, batch_metas

    # 返回生成器的长度
    def __len__(self):
        return int(np.ceil(float(len(self.img_list))/self.batch_size))  

    # 判断是不是图片
    def _is_img(self, ext):
        # 文件名后缀转小写
        ext = ext.lower()
        if ext == '.jpg' or ext== '.png' or ext== '.jpeg':
            return True
        else:
            return False

    # 读取所有图片
    def _get_all_images(self, dataset_dir):
        img_list = []
        for image in os.listdir(dataset_dir):
            # 完整路径
            image_dir = os.path.join(dataset_dir,image)
            # 如果是图片文件
            if self._is_img(os.path.splitext(image_dir)[1]):
                img_list.append(image_dir)
        return np.array(img_list)

    # 在epoch末尾
    def on_epoch_end(self):
        # 打乱数据索引
        if self.shuffle: 
            np.random.shuffle(self.indices)
    
    # 返回数据集大小
    def size(self):
        return len(self.img_list) 
