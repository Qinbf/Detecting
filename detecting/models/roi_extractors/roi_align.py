import tensorflow as tf

from detecting.utils.misc import *

# ROIAlign计算层
class ROIAlign(tf.keras.layers.Layer):
    def __init__(self, pool_shape, **kwargs):
        '''Implements ROI Pooling.

        Attributes
        ---
            pool_shape: (height, width) of the output pooled regions.
                Example: (14, 14)
        '''
        super(ROIAlign, self).__init__(**kwargs)
        self.pool_shape = pool_shape
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name='roi_max_pool')

    # 得到候选区域rois对应的特征图经过ROIAlign计算后的结果
    def __call__(self, inputs):
        '''
        Args
        ---
            rois: [batch_size * num_rois, (batch_ind, y1, x1, y2, x2)] in normalized coordinates.
            feature_map_list: List of [batch_size, height, width, channels].
                feature maps from different levels of the pyramid.
            img_metas: [batch_size, 12]
        ---
            rois: 正负样本的候选框
            feature_map_list: 特征图
            img_metas: 图像元数据
        Returns
        ---
            pooled_rois_list: list of [batch_size * num_rois, pooled_height, pooled_width, channels].
                The width and height are those specific in the pool_shape in the layer
                constructor.
        '''
        # 输入
        rois, feature_map_list, img_metas = inputs
        # 得到候选框的batch_ind
        roi_indices = tf.cast(rois[:, 0], tf.int32)
        # 得到候选框的坐标
        rois = rois[:, 1:]
        # 不计算梯度
        rois = tf.stop_gradient(rois)
        # 使用tf.image.crop_and_resize来近似完成ROIAlign
        # 得到数据的shape为(batch * num_rois, pool_height, pool_width, channels)
        # 假设一个批次有2张图片：
        # feature_map_list[0] = [feature_map1, feature_map2]
        # rois = [boxes1, boxes2, boxes3, boxes4]
        # roi_indices = [ 0, 0, 1, 1]
        # self.pool_shape默认值为14x14
        pooled_rois = tf.image.crop_and_resize(
            feature_map_list[0], rois, roi_indices, self.pool_shape,
            method="bilinear")

        # 返回ROIAlign计算后的结果，数据shape为
        # (batch_size * num_rois, pooled_height, pooled_width, channels)
        return self.max_pool(pooled_rois)