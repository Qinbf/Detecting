import tensorflow as tf

from detecting.utils.misc import *

# 计算候选框与真实标注框的(dy,dx,dh,dw)误差
# target_stds=(0.1, 0.1, 0.2, 0.2)
def bbox2delta(box, gt_box, target_means, target_stds):
    '''Compute refinement needed to transform box to gt_box.
    
    Args
    ---
        box: [..., (y1, x1, y2, x2)]
        gt_box: [..., (y1, x1, y2, x2)]
        target_means: [4]
        target_stds: [4]
    '''
    target_means = tf.constant(
        target_means, dtype=tf.float32)
    target_stds = tf.constant(
        target_stds, dtype=tf.float32)
    # 候选框
    box = tf.cast(box, tf.float32)
    # 真实标注框
    gt_box = tf.cast(gt_box, tf.float32)
    # 候选框高度
    height = box[..., 2] - box[..., 0] 
    # 候选框宽度
    width = box[..., 3] - box[..., 1] 
    # 候选框中心点y值
    center_y = box[..., 0] + 0.5 * height
    # 候选框中心点x值
    center_x = box[..., 1] + 0.5 * width
    # 真实标注框高度
    gt_height = gt_box[..., 2] - gt_box[..., 0] 
    # 真实标注框宽度
    gt_width = gt_box[..., 3] - gt_box[..., 1] 
    # 真实标注框中心点y值
    gt_center_y = gt_box[..., 0] + 0.5 * gt_height
    # 真实标注框中心点x值
    gt_center_x = gt_box[..., 1] + 0.5 * gt_width
    # y方向平移误差
    dy = (gt_center_y - center_y) / height
    # x方向平移误差
    dx = (gt_center_x - center_x) / width
    # 高度缩放误差
    dh = tf.math.log(gt_height / height)
    # 宽度缩放误差
    dw = tf.math.log(gt_width / width)
    # 误差数据堆叠
    delta = tf.stack([dy, dx, dh, dw], axis=-1)
    # 误差数据标准化
    delta = (delta - target_means) / target_stds
    # 返回误差数据
    return delta

# 根据回归预测值delta对anchors或候选框进行调整
def delta2bbox(box, delta, target_means, target_stds):
    '''Compute bounding box based on roi and delta.
    
    Args
    ---
        box: [N, (y1, x1, y2, x2)] box to update
        delta: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        target_means: [4]
        target_stds: [4]
    '''
    target_means = tf.constant(
        target_means, dtype=tf.float32)
    target_stds = tf.constant(
        target_stds, dtype=tf.float32)
    # # 模型预测回归值乘target_stds标准差加target_means均值，反向标准化处理
    delta = delta * target_stds + target_means    
    # 计算的高度
    height = box[:, 2] - box[:, 0] 
    # 计算的宽度
    width = box[:, 3] - box[:, 1] 
    # 计算中心点y坐标
    center_y = box[:, 0] + 0.5 * height
     # 计算中心点x坐标
    center_x = box[:, 1] + 0.5 * width
    
    # 对进行调整
    # y方向平移
    center_y += delta[:, 0] * height
    # x方向平移
    center_x += delta[:, 1] * width
    # 高度缩放
    height *= tf.exp(delta[:, 2])
    # 宽度缩放
    width *= tf.exp(delta[:, 3])
    
    # Convert back to y1, x1, y2, x2
    # 对anchors调节后再变成左上角的坐标和右下角的坐标
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    # 进行堆叠
    result = tf.stack([y1, x1, y2, x2], axis=1)
    return result

# 把调整后的anchors限制在图片范围内
def bbox_clip(box, window):
    '''
    Args
    ---
        box: [N, (y1, x1, y2, x2)]
        window: [4] in the form y1, x1, y2, x2
    '''
    # split数据
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(box, 4, axis=1)
    # 限制y1, x1, y2, x2数值范围，不能超出图片大小
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    # 堆叠数据
    clipped = tf.concat([y1, x1, y2, x2], axis=1)
    clipped.set_shape((clipped.shape[0], 4))
    return clipped

# 预测框翻转
def bbox_flip(bboxes, width):
    '''Flip bboxes horizontally.
    
    Args
    ---
        bboxes: [..., 4]
        width: Int or Float
    '''
    # 切分数据
    y1, x1, y2, x2 = tf.split(bboxes, 4, axis=-1)
    # 计算翻转对应的x1
    new_x1 = width - x2
    # 计算翻转对应的x2
    new_x2 = width - x1
    # 拼接数据
    flipped = tf.concat([y1, new_x1, y2, new_x2], axis=-1)
    
    return flipped


# 从检测框进行缩放
def bbox_mapping(box, img_meta):
    '''
    Args
    ---
        box: [N, 4]
        img_meta: [12]
    '''
    # 图片元数据解析
    img_meta = parse_image_meta(img_meta)
    # 图片缩放因子
    scale = img_meta['scale']
    # 图片是否翻转
    flip = img_meta['flip']
    # 对检测框进行缩放
    # box = box * scale
    box[:, 0] = box[:, 0] * scale[0]
    box[:, 1] = box[:, 1] * scale[1]
    box[:, 2] = box[:, 2] * scale[0]
    box[:, 3] = box[:, 3] * scale[1]
    # 如果需要翻转
    if tf.equal(flip, 1):
        # 对检测框进行翻转
        box = bbox_flip(box, img_meta['img_shape'][1])
    
    return box

# 把预测框映射到原始图片中
def bbox_mapping_back(box, img_meta):
    '''
    Args
    ---
        box: [N, 4]
        img_meta: [12]
    '''
    
    # resize后图片shape
    img_shape = img_meta[..., 3:6]
    # 图片缩放因子
    scale = img_meta[..., 9:11]
    # 图片是否翻转
    flip = img_meta[..., 11]
    # 如果图片翻转
    if tf.equal(flip, 1):
        # 预测框翻转
        box = bbox_flip(box, img_shape[1])
        
    box = box.numpy()
    # 预测框除以缩放因子
    box[:, 0] = box[:, 0] / scale[0]
    box[:, 1] = box[:, 1] / scale[1]
    box[:, 2] = box[:, 2] / scale[0]
    box[:, 3] = box[:, 3] / scale[1]
    
    return box