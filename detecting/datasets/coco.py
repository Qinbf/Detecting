import os.path as osp
import cv2
import numpy as np
from pycocotools.coco import COCO

from detecting.datasets import transforms, utils

class CocoDataSet(object):
    def __init__(self, dataset_dir, subset,
                 flip_ratio=0,
                 pad_mode='fixed',
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 scale=(1024, 800),
                 keep_aspect = False,
                 image_dir = '',
                 label_dir = '',
                 debug=False):
        '''Load a subset of the COCO dataset.
        
        Attributes
        ---
            dataset_dir: The root directory of the COCO dataset.
            subset: What to load (train, val).
            flip_ratio: Float. The ratio of flipping an image and its bounding boxes.
            pad_mode: Which padded method to use (fixed, non-fixed)
            mean: Tuple. Image mean.
            std: Tuple. Image standard deviation.
            scale: Tuple of two integers.
        ---
            dataset_dir: COCO数据集存放位置
            subset: 载入训练集还是验证集(train, val)
            flip_ratio: 图片翻转的概率
            pad_mode: 哪一种padded method，(fixed, non-fixed)
            mean: 图片均值
            std: 图片标准差
            scale: 图片大小
        '''
        # subset必须为['train', 'val', 'test']
        if subset not in ['train', 'val', 'test']:
            raise AssertionError('subset must be "train" "val" or "test".')
        if not image_dir:
            # 图片保存路径
            self.image_dir = "{}/{}2017".format(dataset_dir, subset)
        else:
            self.image_dir = image_dir
        if not label_dir:
            # 标签保存路径
            self.label_dir = '{}/annotations/'.format(dataset_dir)
        else:
            self.label_dir = label_dir
            
        # 获得标签数据
        self.coco = COCO(self.label_dir + 'instances_{}2017.json'.format(subset))
        
        # 得到一个长度为80的list，里面保存各个种类的id号（注意并不是从1-80，而是从1-90，中间有些编号空缺）
        self.cat_ids = self.coco.getCatIds()
        # 得到一个cat2label字典，键为种类的id号，值为1-80编号
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        # 得到一个label2cat字典，键为1-80编号，值为种类的id号
        self.label2cat = {
            i + 1: cat_id
            for i, cat_id in enumerate(self.cat_ids)
        }
        # 得到保存图片id的list和保存图片信息的list
        self.img_ids, self.img_infos = self._filter_imgs()
        # debug取前20张图片
        if debug:
            self.img_ids, self.img_infos = self.img_ids[:20], self.img_infos[:20]
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
        
    # 图片过滤器
    def _filter_imgs(self, min_size=32):
        '''Filter images too small or without ground truths.
        
        Args
        ---
            min_size: the minimal size of the image.
        '''
        # 获得所有图片的id
        all_img_ids = list(set([_['image_id'] for _ in self.coco.anns.values()]))
        # 用于保存图片id
        img_ids = []
        # 用于保存图片信息
        img_infos = []
        # 循环所有图片的id
        for i in all_img_ids:
            # 获得i图片的信息
            info = self.coco.loadImgs(i)[0]
            # 获得i图片的标注id号
            ann_ids = self.coco.getAnnIds(imgIds=i)
            # 根据标注id获得标注信息
            ann_info = self.coco.loadAnns(ann_ids)
            # 标注框处理
            ann = self._parse_ann_info(ann_info)
            # 如果图片长宽都大于最小size，并且ann中有标签值
            if min(info['width'], info['height']) >= min_size and ann['labels'].shape[0] != 0:
                # 保存该图片
                img_ids.append(i)
                # 保存该图片的信息
                img_infos.append(info)
        # 返回保存图片id的list和保存图片信息的list
        return img_ids, img_infos
        
    # 获得图片标签信息
    def _load_ann_info(self, idx):
        # 获得第idx张图片
        img_id = self.img_ids[idx]
        # 获得该图片的标注id号
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # 根据标注id获得标注信息
        ann_info = self.coco.loadAnns(ann_ids)
        # 返回标注信息
        return ann_info

    # 标注框处理
    def _parse_ann_info(self, ann_info):
        '''Parse bbox annotation.
        
        Args
        ---
            ann_info (list[dict]): Annotation info of an image.
            
        Returns
        ---
            dict: A dict containing the following keys: bboxes, 
                bboxes_ignore, labels.
        '''
        # 保存单个对象标注框坐标
        gt_bboxes = []
        # 保存单个对象标注框标签
        gt_labels = []
        # 保存多个对象标注框坐标
        gt_bboxes_ignore = []
        # 循环标注信息
        for i, ann in enumerate(ann_info):
            # 查询ignore，如果不存在则返回False
            # 如果存在ignore，则跳过该标注
            if ann.get('ignore', False):
                continue
            # 获得标注框的坐标，x1和y1为左上角坐标值，w为标注框宽度，h为标注框高度
            x1, y1, w, h = ann['bbox']
            # 如果标注框面积，宽度，高度存在异常则跳过该标注
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            # 标注框左上角坐标和右下角坐标
            bbox = [y1, x1, y1 + h - 1, x1 + w - 1]
            # 如果为一组对象
            if ann['iscrowd']:
                # 保存标注框坐标
                gt_bboxes_ignore.append(bbox)
            # 如果为单个对象
            else:
                # 保存标注框坐标
                gt_bboxes.append(bbox)
                # 保存标注框对应标签
                gt_labels.append(self.cat2label[ann['category_id']])
        # 定义数据类型
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        # 定义数据类型
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        # 把gt_bboxes，gt_labels，gt_bboxes_ignore存入字典返回
        ann = dict(bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)
        return ann
    
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
        # 获得图片标注信息
        ann_info = self._load_ann_info(idx)
        
        # 读取图片
        img = cv2.imread(osp.join(self.image_dir, img_info['file_name']), cv2.IMREAD_COLOR)
        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 获得图片shape
        ori_shape = img.shape
        
        # 标注框处理
        ann = self._parse_ann_info(ann_info)
        # 获得单个对象的标注框坐标
        bboxes = ann['bboxes']
        # 获得单个对象的标注框标签
        labels = ann['labels']
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
        '''Get list of category names. 
        
        Returns
        ---
            list: A list of category names.
            
        Note that the first item 'bg' means background.
        '''
        return ['background'] + [self.coco.loadCats(i)[0]["name"] for i in self.cat2label.keys()]

    # 把category编号变成label编号
    def cat_label(self, class_ids):
        label = []
        for cat in class_ids:
            label.append(self.cat2label[cat])
        return np.array(label)