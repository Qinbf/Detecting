import numpy as np

from detecting.datasets.utils import *

class ImageTransform(object):
    '''Preprocess the image.
    
        1. rescale the image to expected size
        2. normalize the image
        3. flip the image (if needed)
        4. pad the image (if needed)
    ---
        1.rezise图片大小
        2.图片标准化处理
        3.图片翻转
        4.图片填充
    '''
    def __init__(self,
                 scale=(800, 1333),
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 pad_mode='fixed',
                 keep_aspect=False):
        # 图片大小
        self.scale = scale
        # 图片均值
        self.mean = mean
        # 图片标准差
        self.std = std
        # pad模式
        self.pad_mode = pad_mode
        # 图片pad大小
        self.impad_size = max(scale) if pad_mode == 'fixed' else 64
        # 是否保持长宽比
        self.keep_aspect = keep_aspect

    def __call__(self, img, flip=False):
        # resize图片，得到reszie后的图片和缩放因子
        img, scale_factor = imrescale(img, self.scale, self.keep_aspect)
        # 获得图片shape
        img_shape = img.shape
        # 图片标准化处理
        img = imnormalize(img, self.mean, self.std)
        # 如果图片翻转
        if flip:
            img = img_flip(img)
        # 图片使用fixed模式填充
        if self.pad_mode == 'fixed':
            img = impad_to_square(img, self.impad_size)
        # 图片使用non-fixed模式填充
        else: 
            img = impad_to_multiple(img, self.impad_size)
        # 返回填充后的图片，resize后的图片shape，缩放因子
        return img, img_shape, scale_factor

class BboxTransform(object):
    '''Preprocess ground truth bboxes.
    
        1. rescale bboxes according to image size
        2. flip bboxes (if needed)
    ---
        1.resize标注框
        2.标注框翻转
    '''
    def __init__(self):
        pass
    
    def __call__(self, bboxes, labels, 
                 img_shape, scale_factor, flip=False):
        trans_bboxes = np.zeros(bboxes.shape, dtype=np.float32)
        # 标注框坐标乘以缩放因子
        trans_bboxes[:, 0] = bboxes[:, 0] * scale_factor[0]
        trans_bboxes[:, 1] = bboxes[:, 1] * scale_factor[1]
        trans_bboxes[:, 2] = bboxes[:, 2] * scale_factor[0]
        trans_bboxes[:, 3] = bboxes[:, 3] * scale_factor[1]
        # 如果需要翻转
        if flip:
            trans_bboxes = bbox_flip(trans_bboxes, img_shape)
        # 把标注框坐标限制在图片内
        trans_bboxes[:, 0::2] = np.clip(trans_bboxes[:, 0::2], 0, img_shape[0])
        trans_bboxes[:, 1::2] = np.clip(trans_bboxes[:, 1::2], 0, img_shape[1])
        # 返回处理后的标注框，标签没处理直接返回
        return trans_bboxes, labels
