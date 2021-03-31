import os.path as osp
import cv2
import numpy as np
from detecting.datasets import transforms, utils
import glob
import xml.etree.ElementTree as ET

class VocDataSet(object):
    def __init__(self, dataset_dir, subset,
                 flip_ratio=0,
                 pad_mode='fixed',
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 scale=(1024, 800),
                 keep_aspect = False,
                 classes = '',
                 image_dir = '',
                 label_dir = '',
                 debug=False):
        '''Load a subset of the VOC dataset.
        
        Attributes
        ---
            dataset_dir: The root directory of the VOC dataset.
            subset: What to load (train, test).
            flip_ratio: Float. The ratio of flipping an image and its bounding boxes.
            pad_mode: Which padded method to use (fixed, non-fixed)
            mean: Tuple. Image mean.
            std: Tuple. Image standard deviation.
            scale: Tuple of two integers.
        ---
            dataset_dir: VOC数据集存放位置
            subset: 载入训练集还是验证集(train, test)
            flip_ratio: 图片翻转的概率
            pad_mode: 哪一种padded method，(fixed, non-fixed)
            mean: 图片均值
            std: 图片标准差
            scale: 图片大小
        '''
        # subset必须为['train', 'test']
        if subset not in ['train', 'test']:
            raise AssertionError('subset must be "train" or "test".')
        
        if not classes:
            # 20个种类+背景
            self._classes = ('__background__',  # always index 0
                            'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog', 'horse',
                            'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor')
        else:
            self._classes = classes
        if not image_dir:
            # 图片保存路径
            self.image_dir = "{}/{}/VOC2007/JPEGImages".format(dataset_dir, subset)
        else:
            self.image_dir = image_dir
        if not label_dir:
            # 标签保存路径
            self.label_dir = "{}/{}/VOC2007/Annotations".format(dataset_dir, subset)
        else:
            self.label_dir = label_dir

        # 物体名称对应编号的字典
        self._class_to_ind = dict(zip(self._classes, range(len(self._classes))))     
        # 获得图片标签信息
        self.img_infos = self._load_ann_info(self.label_dir)
        # 图片翻转的概率
        self.flip_ratio = flip_ratio
        # pad模式
        if pad_mode in ['fixed', 'non-fixed']:
            self.pad_mode = pad_mode
        elif subset == 'train':
            self.pad_mode = 'fixed'
        else:
            self.pad_mode = 'non-fixed'

        self.keep_aspect = keep_aspect
        # ImageTransform用于处理图片
        self.img_transform = transforms.ImageTransform(scale, mean, std, pad_mode, self.keep_aspect)
        # BboxTransform用于处理标签
        self.bbox_transform = transforms.BboxTransform()
        
    # 获得图片标签信息
    def _load_ann_info(self, ann_path, min_size=32):
        # 用于保存图片信息
        img_infos = []
        # 循环路径下所有xml文件
        for xml_file in glob.glob(ann_path + '/*.xml'):
            # 保存标注框坐标
            gt_bboxes = []
            # 保存标注框标签
            gt_labels = []
            # xml文件解析
            tree = ET.parse(xml_file)
            root = tree.getroot()
            # 图片名
            filename = root.find('filename').text
            # 图片宽度
            width = int(root.find('size')[0].text)
            # 图片高度
            height = int(root.find('size')[1].text)
            # 图片通道数
            depth = int(root.find('size')[2].text)
            # 图片shape为(height,width,depth)
            shape = (height,width,depth)
            # 循环该图片中的标注
            for member in root.findall('object'):
                # 类别名称
                class_name = member.find('name').text
                # 坐标
                x1 = float(member.find('bndbox')[0].text)
                y1 = float(member.find('bndbox')[1].text)
                x2 = float(member.find('bndbox')[2].text)
                y2 = float(member.find('bndbox')[3].text)    
                           
                # 计算标注框框高度
                h = y2 - y1
                # 计算标注框宽度
                w = x2 - x1
                # 如果宽度，高度存在异常则跳过该标注
                if w < 1 or h < 1:
                    continue
                # 标注框左上角坐标和右下角坐标
                bbox = [y1, x1, y2, x2]
                # 保存标注框坐标
                gt_bboxes.append(bbox)
                # 保存标注框对应标签
                gt_labels.append(self._class_to_ind[class_name])
            # 定义数据类型
            if gt_bboxes:
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                gt_labels = np.array(gt_labels, dtype=np.int64)
            else:
                gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.array([], dtype=np.int64)
            
            # 把file_name,gt_bboxes,gt_labels存入字典
            info = dict(file_name=filename,
                        width=width,
                        height=height,
                        depth=depth,
                        shape=shape,
                        bboxes=gt_bboxes,
                        labels=gt_labels)
            # 如果图片长宽都大于最小size，并且ann中有标签值
            if min(info['width'], info['height']) >= min_size and info['labels'].shape[0] != 0:
                # 保存该图片的信息
                img_infos.append(info)
        return img_infos

    # 返回图片数量
    def __len__(self):
        return len(self.img_infos)
    
    # 用于实现迭代功能
    def __getitem__(self, idx):
        '''Load the image and its bboxes for the given index.
        
        Args
        ---
            idx: the index of images.
            
        Returns
        ---
            tuple: A tuple containing the following items: image, 
                bboxes, labels.
        '''
        # 图片信息
        img_info = self.img_infos[idx]      
        # 读取图片
        img = cv2.imread(osp.join(self.image_dir, img_info['file_name']), cv2.IMREAD_COLOR)
        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 获得图片shape
        ori_shape = img.shape
        # 获得单个对象的标注框坐标
        bboxes = img_info['bboxes']
        # 获得单个对象的标注框标签
        labels = img_info['labels']
        # 是否进行翻转
        flip = True if np.random.rand() < self.flip_ratio else False
        
        # 处理图片数据
        # 得到填充后的图片，resize后图片shape，缩放因子
        img, img_shape, scale_factor = self.img_transform(img, flip)
        # 填充后的图片shape
        pad_shape = img.shape
        
        # 处理标注框数据
        trans_bboxes, labels = self.bbox_transform(
            bboxes, labels, img_shape, scale_factor, flip)
        
        # 保存原始图片shape
        # resize后的图片shape
        # 填充后的图片shape
        # 缩放因子
        # 是否翻转
        img_meta_dict = dict({
            'ori_shape': ori_shape,
            'img_shape': img_shape,
            'pad_shape': pad_shape,
            'scale_factor': scale_factor,
            'flip': flip
        })
        # 把img_meta_dict中的数据组成1维的array
        img_meta = utils.compose_image_meta(img_meta_dict)
        # 返回处理后的图片数据，图片相关的一些信息，图片中的标注框坐标，标注框标签
        return img, img_meta, trans_bboxes, labels
    
    # 获得所有类别的名称
    def get_categories(self):
        return self._classes

