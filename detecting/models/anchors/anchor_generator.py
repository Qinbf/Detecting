import tensorflow as tf

from detecting.utils.misc import *

# 生成anchor
class AnchorGenerator(object):
    def __init__(self, 
                 scales=(32, 64, 128, 256, 512), 
                 ratios=(0.5, 1, 2), 
                 feature_strides=(4, 8, 16, 32, 64)):
        '''Anchor Generator
        
        Attributes
        ---
            scales: 1D array of anchor sizes in pixels.
            ratios: 1D array of anchor ratios of width/height.
            feature_strides: Stride of the feature map relative to the image in pixels.
        '''
        # anchor大小
        self.scales = scales
        # anchor比例
        self.ratios = ratios
        # 特征步长
        self.feature_strides = feature_strides


    # 得到anchors以及anchors是否有效的标注
    def generate_anchors(self, img_metas, feature_shape):
        '''Generate the anchors for Region Proposal Network
        
        Args
        ---
            feature_shape: [feat_map_height, feat_map_width]
        Returns
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
        '''

        # 通过feature_shape生成anchors
        anchors = [self._generate_anchors(0, feature_shape)]
  
        # 合并anchors
        anchors = tf.concat(anchors, axis=0)

        # generate valid flags
        # 得到resize后的图片shape
        img_shapes = calc_img_shapes(img_metas)
        # for i in range(img_shapes.shape[0])为了循环一个批次的数据
        # anchors是通过填充后的图片产生的特征图生成的，所以会有一些无效的anchors
        # 得到有效anchors的标注
        valid_flags = [
            self._generate_valid_flags(anchors, img_shapes[i])
            for i in range(img_shapes.shape[0])
        ]
        # 堆叠一个批次的valid_flags
        valid_flags = tf.stack(valid_flags, axis=0)
        # 返回anchors和valid_flags
        return anchors, valid_flags

    
    # 得到有效anchors的标注
    def _generate_valid_flags(self, anchors, img_shape):
        '''
        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            img_shape: Tuple. (height, width, channels)
            
        Returns
        ---
            valid_flags: [num_anchors]
        '''
        # 根据anchors的数量，生成全为1的数
        valid_flags = tf.ones(anchors.shape[0], dtype=tf.int32)
        # 根据anchors的数量，生成全为0的数
        zeros = tf.zeros(anchors.shape[0], dtype=tf.int32)
        # anchors在图片范围内的标注为有效anchors
        valid_flags = tf.where((anchors[:, 0]>=0) &
                              (anchors[:, 1]>=0) &
                              (anchors[:, 2]<img_shape[0]) &
                              (anchors[:, 3]<img_shape[1]),
                              valid_flags, zeros)

        # 返回anchors的标注
        return valid_flags
    
    # 产生anchors
    def _generate_anchors(self, level, feature_shape):
        '''Generate the anchors given the spatial shape of feature map.
        
        Args
        ---
            feature_shape: (height, width)

        Returns
        ---
            numpy.ndarray [anchors_num, (y1, x1, y2, x2)]
        '''
        # 比如有4个尺度(64,128,256,512)
        scale = tf.cast(tf.convert_to_tensor(self.scales), dtype=tf.float32)
        # 比如每个尺度都有3种比例(0.5,1,2)
        ratios = self.ratios
        # 该尺度的步长,比如为16
        feature_stride = self.feature_strides[level]
        
        # scales-[64,128,256,512,64,128,256,512,64,128,256,512]
        # ratios-[[0.5,0.5,0.5,0.5],
        #         [1,  1,  1,  1],
        #         [2,  2,  2,  2]]
        scales, ratios = tf.meshgrid(scale, ratios)
        # 变成1维
        scales = tf.reshape(scales, [-1])
        ratios = tf.reshape(ratios, [-1])
        
        # anchor的12种高度
        heights = scales / tf.sqrt(ratios)
        # anchor的12种宽度
        widths = scales * tf.sqrt(ratios) 

        # 比如feature_shape的值为(64,64)
        # 64个y值
        shifts_y = tf.multiply(tf.range(feature_shape[0]), feature_stride)
        # 64个x值
        shifts_x = tf.multiply(tf.range(feature_shape[1]), feature_stride)
        # 转为float类型
        shifts_x, shifts_y = tf.cast(shifts_x, tf.float32), tf.cast(shifts_y, tf.float32)
        # 产生网格点阵
        # shifts_x和shifts_x的shape都是(64,64)
        shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y)

        # 4096个中心点每个点都对应12个宽度和12个高度
        # box_widths,box_centers_x,box_heights,box_centers_y的shape都为(4096,12)
        box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)

        # tf.stack堆叠box_centers_y, box_centers_x得到的数据shape为(4096,12,2)
        # 然后再reshape为(49152, 2)
        # 注意这里的49152代表49152个anchor，特征图上有4096个点，每个特征图上的点对应12个anchor，2代表anchor的中心点位置
        box_centers = tf.reshape(tf.stack([box_centers_y, box_centers_x], axis=2), (-1, 2))
        # tf.stack堆叠box_heights, box_widths得到的数据shape为(4096,12,2)
        # 然后再reshape为(4096,2)
        # 注意这里的49152代表49152个anchor，特征图上有4096个点，每个特征图上的点对应12个anchor，2代表anchor的中心点位置
        box_sizes = tf.reshape(tf.stack([box_heights, box_widths], axis=2), (-1, 2))
        
        # 计算每个anchor的左上角和右下角坐标
        # box_centers - 0.5 * box_sizes左上角坐标
        # box_centers + 0.5 * box_sizes右下角坐标
        boxes = tf.concat([box_centers - 0.5 * box_sizes,
                           box_centers + 0.5 * box_sizes], axis=1)

        # 返回anchor，shape为(49152, 4)
        return boxes
