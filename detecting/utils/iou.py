import tensorflow as tf

# 计算预测框和真实标注的交并比
def compute_overlaps(boxes1, boxes2):
    '''Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    '''
    # boxes1预测框，shape为(2000,4)
    # boxes2真实标注框，假设shape为(6,4)
    # tf.tile用于复制扩张数据
    # 第0个维是原来的1倍，第1维是原来的1倍，第2维是原来的6倍(2000,1,4)->(2000,1,24)
    # reshape(2000,1,24)->(12000,4)
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    # 第0维是原来的2000倍，(6,4)->(12000,4)
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 各个坐标值切分出来得到数据shape为(12000,1)
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    # 计算相交区域左上角的坐标和右下角的坐标
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    # 交集面积
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 计算预测框面积
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    # 计算真实标注框面积
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    # 计算并集面积
    union = b1_area + b2_area - intersection
    # 计算交并比IOU
    iou = intersection / union
    # reshape为(2000,6)
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    # 返回交并比结果
    return overlaps
