import cv2
import numpy as np
import tensorflow as tf

###########################################
#
# Utility Functions for 
# Image Preprocessing and Data Augmentation
#
###########################################
# 图片翻转
def img_flip(img):
    '''Flip the image horizontally
    
    Args
    ---
        img: [height, width, channel]
    
    Returns
    ---
        np.ndarray: the flipped image.
    '''
    return np.fliplr(img)

# 标注框翻转
def bbox_flip(bboxes, img_shape):
    '''Flip bboxes horizontally.
    
    Args
    ---
        bboxes: [..., 4]
        img_shape: Tuple. (height, width)
    
    Returns
    ---
        np.ndarray: the flipped bboxes.
    '''
    # 获得图片宽度
    w = img_shape[1]
    flipped = bboxes.copy()
    # 这里的flipped[..., 1]等于flipped[:, 1]
    # 如果flipped是3维数据flipped[..., 1]等于flipped[:,:,1]
    # 标注框坐标水平翻转
    flipped[..., 1] = w - bboxes[..., 3] - 1
    flipped[..., 3] = w - bboxes[..., 1] - 1
    return flipped

# 图片填充
def impad_to_square(img, pad_size):
    '''Pad an image to ensure each edge to equal to pad_size.
    
    Args
    ---
        img: [height, width, channels]. Image to be padded
        pad_size: Int.
    
    Returns
    ---
        ndarray: The padded image with shape of 
            [pad_size, pad_size, channels].
    '''
    # 填充后的图片shape
    shape = (pad_size, pad_size, img.shape[-1])
    # 定义一个值为0的图像数据
    pad = np.zeros(shape, dtype=img.dtype)
    # 把原图片的数值复制过来
    pad[:img.shape[0], :img.shape[1], ...] = img
    # 返回填充后的图片
    return pad

# 图片填充
def impad_to_multiple(img, divisor):
    '''Pad an image to ensure each edge to be multiple to some number.
    
    Args
    ---
        img: [height, width, channels]. Image to be padded.
        divisor: Int. Padded image edges will be multiple to divisor.
    
    Returns
    ---
        ndarray: The padded image.
    ''' 
    # 把图片高度变成divisor的倍数
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    # 把图片宽度变成divisor的倍数
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    # 定义新的图片形状
    shape = (pad_h, pad_w, img.shape[-1])
    # 定义一个值为0的图像数据
    pad = np.zeros(shape, dtype=img.dtype)
    # 把原图片的数值复制过来
    pad[:img.shape[0], :img.shape[1], ...] = img
    # 返回填充后的图片
    return pad

# resize图片
def imrescale(img, scale, keep_aspect):
    '''Resize image 
    
    Args
    ---
        img: [height, width, channels]. The input image.
        scale: Tuple of 2 integers. the image will be rescaled 
            as large as possible within the scale
    
    Returns
    ---
        np.ndarray: the scaled image.
    ''' 
    # 获得图片的高度和宽度
    h, w = img.shape[:2]

    # 保持长宽比
    if keep_aspect == True:
        # 最长的边
        max_long_edge = max(scale)
        # 最短的边
        max_short_edge = min(scale)
        # 缩放因子
        scale_factor = min(max_long_edge / max(h, w),
                    max_short_edge / min(h, w))
        # 长宽都是一样的缩放因子
        scale_factor = (scale_factor,scale_factor)
        # 得到新图片尺寸
        new_size = (int(w * float(scale_factor) + 0.5),
                    int(h * float(scale_factor) + 0.5))
        # resize图片大小
        rescaled_img = cv2.resize(
            img, new_size, interpolation=cv2.INTER_LINEAR)
    else:
        # 缩放因子
        scale_factor = (scale[0] / h,
                        scale[1] / w)
        # resize图片大小
        rescaled_img = cv2.resize(
            img, (scale[1],scale[0]), interpolation=cv2.INTER_LINEAR)
    # 返回resize后的图片和缩放因子
    return rescaled_img, scale_factor

# 图像标准化处理
def imnormalize(img, mean, std):
    '''Normalize the image.
    
    Args
    ---
        img: [height, width, channel]
        mean: Tuple or np.ndarray. [3]
        std: Tuple or np.ndarray. [3]
    
    Returns
    ---
        np.ndarray: the normalized image.
    '''
    img = (img - mean) / std    
    return img.astype(np.float32)

# 反标准化处理
def imdenormalize(norm_img, mean, std):
    '''Denormalize the image.
    
    Args
    ---
        norm_img: [height, width, channel]
        mean: Tuple or np.ndarray. [3]
        std: Tuple or np.ndarray. [3]
    
    Returns
    ---
        np.ndarray: the denormalized image.
    '''
    img = norm_img * std + mean
    return img.astype(np.float32)

#######################################
#
# Utility Functions for Data Formatting
#
#######################################
# 获得原始图片，一个解析一张图片
def get_original_image(img, img_meta, 
                       mean=(123.68,116.779,103.939), std=(1, 1, 1)):
    '''Recover the origanal image.
    
    Args
    ---
        img: np.ndarray. [height, width, channel]. 
            The transformed image.
        img_meta: np.ndarray. [12]
        mean: Tuple or np.ndarray. [3]
        std: Tuple or np.ndarray. [3]
    
    Returns
    ---
        np.ndarray: the original image.
    '''
    if img_meta.ndim == 2:
        img_meta = img_meta[0]
    if img.ndim == 4:
        img = img[0]
    # 解析图片元数据
    img_meta_dict = parse_image_meta(img_meta)
    # 图片原始shape
    ori_shape = img_meta_dict['ori_shape']
    # 图片resize后的shape
    img_shape = img_meta_dict['img_shape']
    # 图片是否翻转
    flip = img_meta_dict['flip']
    # 从填充后的图片中获取resize后的图片大小
    img = img[:img_shape[0], :img_shape[1]]
    # 如果图片翻转，重新翻转图片
    if flip:
        img = img_flip(img)
    # 把图片resize变成原始大小
    img = cv2.resize(img, (ori_shape[1], ori_shape[0]), 
                     interpolation=cv2.INTER_LINEAR)
    # 反标准化处理
    img = imdenormalize(img, mean, std)
    # 返回图片
    return img

# 把图像相关的一些数据组成一维的array
def compose_image_meta(img_meta_dict):
    '''Takes attributes of an image and puts them in one 1D array.

    Args
    ---
        img_meta_dict: dict

    Returns
    ---
        img_meta: np.ndarray
    '''
    # 原始图片shape
    ori_shape = img_meta_dict['ori_shape']
    # resize后的图片shape
    img_shape = img_meta_dict['img_shape']
    # 填充后的图片shape
    pad_shape = img_meta_dict['pad_shape']
    # 缩放因子
    scale_factor = img_meta_dict['scale_factor']
    # 是否翻转
    flip = 1 if img_meta_dict['flip'] else 0
    # 组成一个12个值的array
    img_meta = np.array(
        ori_shape +               # size=3
        img_shape +               # size=3
        pad_shape +               # size=3
        scale_factor +            # size=2
        tuple([flip])             # size=1
    ).astype(np.float32)
    # 返回
    return img_meta

# 解析图片元数据
def parse_image_meta(img_meta):
    '''Parses an array that contains image attributes to its components.

    Args
    ---
        meta: [12]

    Returns
    ---
        a dict of the parsed values.
    '''
    # 图片原始shape
    ori_shape = img_meta[0:3]
    # 图片resize后的shape
    img_shape = img_meta[3:6]
    # 图片填充后的shape
    pad_shape = img_meta[6:9]
    # 图片缩放因子
    scale_factor = img_meta[9:11]
    # 图片是否翻转
    flip = img_meta[11]
    return {
        'ori_shape': ori_shape.astype(np.int32),
        'img_shape': img_shape.astype(np.int32),
        'pad_shape': pad_shape.astype(np.int32),
        'scale_factor': scale_factor.astype(np.float32),
        'flip': flip.astype(np.bool),
    }

# 读取一张图片
def load_img(img_dir):
    # 读取图片
    img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    # BGR转RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
