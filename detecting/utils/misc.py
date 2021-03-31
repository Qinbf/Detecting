import tensorflow as tf

# 用于删除数值都为0的候选框
def trim_zeros(boxes, name=None):
    '''Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.
    
    Args
    ---
        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
    '''
    # 对候选框数值求和
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    # 只留下数值不为0的候选框
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    # 返回数值不为0的候选框
    return boxes, non_zeros

# 图片元数据解析
def parse_image_meta(meta):
    '''Parses a tensor that contains image attributes to its components.
    
    Args
    ---
        meta: [..., 12]

    Returns
    ---
        a dict of the parsed tensors.
    '''
    meta = meta.numpy()
    # 原始图片shape
    ori_shape = meta[..., 0:3]
    # resize后图片shape
    img_shape = meta[..., 3:6]
    # 填充后图片shape
    pad_shape = meta[..., 6:9]
    # 图片缩放因子
    scale = meta[..., 9:11]
    # 图片是否翻转
    flip = meta[..., 11]
    return {
        'ori_shape': ori_shape,
        'img_shape': img_shape,
        'pad_shape': pad_shape,
        'scale': scale,
        'flip': flip
    }

# 计算一个批次中填充后的图片的最大高度和宽度
def calc_batch_padded_shape(meta):
    '''
    Args
    ---
        meta: [batch_size, 12]
    
    Returns
    ---
        nd.ndarray. Tuple of (height, width)
    '''
    # meta[:, 6:8]填充后的图片shape
    # tf.reduce_max计算最大值
    return tf.cast(tf.reduce_max(meta[:, 6:8], axis=0), tf.int32).numpy()

# 得到resize后的图片shape
def calc_img_shapes(meta):
    '''
    Args
    ---
        meta: [..., 12]
    
    Returns
    ---
        nd.ndarray. [..., (height, width)]
    '''
    # meta[:, 3:5]resize后的图片shape
    return tf.cast(meta[..., 3:5], tf.int32).numpy()

# 得到填充后的图片shape
def calc_pad_shapes(meta):
    '''
    Args
    ---
        meta: [..., 12]
    
    Returns
    ---
        nd.ndarray. [..., (height, width)]
    '''
    # meta[:, 6:8]填充后的图片shape
    return tf.cast(meta[..., 6:8], tf.int32).numpy()