import tensorflow as tf
layers = tf.keras.layers

from detecting.utils import transforms
from detecting.loss import losses
from detecting.utils.misc import *

# 对ROIAlign计算后的候选框特征图数据进行分类回归预测
class BBoxHead(tf.keras.Model):
    def __init__(self, num_classes, 
                 pool_size=7,
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 min_confidence=0.05,
                 nms_threshold=0.5,
                 max_instances=100,
                 head_to_tail=None,
                 share_box_across_classes=True,
                 weight_decay=0,
                 **kwags):
        super(BBoxHead, self).__init__(**kwags)
        
        self.num_classes = num_classes
        self.pool_size = pool_size
        self.target_means = target_means
        self.target_stds = target_stds
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.max_instances = max_instances
        self.head_to_tail = head_to_tail
        self.share_box_across_classes = share_box_across_classes
        self.weight_decay = weight_decay
        # 定义loss
        self.rcnn_class_loss = losses.RCNNClassLoss()
        self.rcnn_bbox_loss = losses.RCNNBBoxLoss()
        
        # 初始化
        init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
        # 正则化
        reg = tf.keras.regularizers.l2(self.weight_decay)
        # 共享回归预测
        self.number_of_boxes = 1
        if not self.share_box_across_classes:
            self.number_of_boxes = self.num_classes

        # 平均池化层
        self.avg_pool = layers.GlobalAveragePooling2D(name='rcnn_avg_pool')

        # 全连接层，用于分类预测
        self.rcnn_class_logits = layers.Dense(self.num_classes, kernel_initializer=init, kernel_regularizer=reg, name='rcnn_class_logits')

        # 全连接层，用于回归预测
        self.rcnn_delta_fc = layers.Dense(self.number_of_boxes * 4, kernel_initializer=init, kernel_regularizer=reg, name='rcnn_bbox_fc')
 

    # 对ROIAlign计算后的候选框特征图数据进行分类回归预测
    def __call__(self, inputs, training=True):
        '''
        Args
        ---
            pooled_rois: [batch_size * num_rois, pool_size, pool_size, channels]
        
        Returns
        ---
            rcnn_class_logits: [batch_size * num_rois, num_classes]
            rcnn_probs: [batch_size * num_rois, num_classes]
            rcnn_deltas: [batch_size * num_rois, (dy, dx, log(dh), log(dw))]
        '''
        pooled_rois = inputs


        # 对ROIAlign计算后的候选框特征图数据进行卷积
        x = self.head_to_tail(pooled_rois)
        if len(x.shape)==4:
            # 平均池化
            x = self.avg_pool(x)

        # 分类预测
        logits = self.rcnn_class_logits(x)
        probs = tf.nn.softmax(logits)
        # 回归预测
        deltas = self.rcnn_delta_fc(x)

        # 返回分类预测结果，分类预测概率值，回归预测
        return logits, probs, deltas

    # 计算RCNN分类回归loss
    def loss(self, 
             rcnn_class_logits, rcnn_deltas, 
             rcnn_labels, rcnn_label_weights, rcnn_delta_targets, rcnn_delta_weights, batch_size):
        '''Calculate RCNN loss
        '''
        # 计算RCNN分类loss
        rcnn_class_loss = self.rcnn_class_loss(
            rcnn_labels, rcnn_class_logits, rcnn_label_weights, batch_size)
        # 计算RCNN回归loss
        rcnn_bbox_loss = self.rcnn_bbox_loss(
            rcnn_delta_targets, rcnn_deltas, rcnn_delta_weights, batch_size)
        # 返回
        return rcnn_class_loss, rcnn_bbox_loss
    
    # 得到rcnn最终预测结果
    def get_bboxes(self, rcnn_probs, rcnn_deltas, rois, img_metas):
        '''
        Args
        ---
            rcnn_probs: [batch_size * num_rois, num_classes]
            rcnn_deltas: [batch_size * num_rois, num_classes, (dy, dx, log(dh), log(dw))]
            rois: [batch_size * num_rois, (batch_ind, y1, x1, y2, x2)]
            img_meta_list: [batch_size, 12]
        ---
            rcnn_probs: rcnn预测分类概率
            rcnn_deltas: rcnn预测回归值
            rois: 候选框数据
            img_meta_list: 图像元数据
        Returns
        ---
            detections_list: List of [num_detections, (y1, x1, y2, x2, class_id, score)]
                coordinates are in pixel coordinates.
        '''
        # 批次大小
        batch_size = img_metas.shape[0]
        # rcnn_probs进行reshape，变成(batch_size,num_rois,num_classes)
        rcnn_probs = tf.reshape(rcnn_probs, (batch_size, -1, self.num_classes))
        # rcnn_deltas进行reshape，变成(batch_size,num_rois,num_classes,4)
        rcnn_deltas = tf.reshape(rcnn_deltas, (batch_size, -1, 4))
        # rois进行reshape，然后再取坐标值，变成(batch_size,num_rois,4)
        rois = tf.reshape(rois, (batch_size, -1, 5))[:, :, 1:5]
        # 得到填充后的图片shape 
        pad_shapes = calc_pad_shapes(img_metas)
        # 循环批次中的数据，得到rcnn最终预测结果
        detections_list = [
            self._get_bboxes_single(
                rcnn_probs[i], rcnn_deltas[i], rois[i], pad_shapes[i])
            for i in range(img_metas.shape[0])
        ]
        # 返回预测结果
        return detections_list
    
    # 得到rcnn最终预测结果
    def _get_bboxes_single(self, rcnn_probs, rcnn_deltas, rois, img_shape):
        '''
        Args
        ---
            rcnn_probs: [num_rois, num_classes]
            rcnn_deltas: [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
            rois: [num_rois, (y1, x1, y2, x2)]
            img_shape: np.ndarray. [2]. (img_height, img_width)  
        ---
            rcnn_probs: rcnn预测分类概率
            rcnn_deltas: rcnn预测回归值
            rois: 候选框数据
            img_shape: 填充后的图片shape 
        '''
        # 图片的高度和宽度
        H, W = img_shape   
        # 得到概率最大的类别的编号
        class_ids = tf.argmax(rcnn_probs, axis=1, output_type=tf.int32)
        # indices
        # 0     24
        # 1     10
        # 2     1
        # ...
        # 1999  45
        indices = tf.stack([tf.range(rcnn_probs.shape[0]), class_ids], axis=1)
        # 取出每个候选框的最大的概率值
        class_scores = tf.gather_nd(rcnn_probs, indices)
        # 取出每个候选框最大概率的类别对应的回归值
        # deltas_specific = tf.gather_nd(rcnn_deltas, indices) 

        # 归一化的坐标值恢复到真实坐标值
        rois *= tf.constant([H, W, H, W], dtype=tf.float32)
        

        # 根据回归预测值对候选框进行调整，得到rcnn预测框
        refined_rois = transforms.delta2bbox(rois, rcnn_deltas, self.target_means, self.target_stds)
        
        
        # 把预测框限制在图片区域范围内
        window = tf.constant([0., 0., H * 1., W * 1.], dtype=tf.float32)
        refined_rois = transforms.bbox_clip(refined_rois, window)
        
        # 筛选出class_ids大于0的类别，也就是排除背景类别
        keep = tf.where(class_ids > 0)[:, 0]
        
        # self.min_confidence默认值为0.05，这个参数一般需要根据情况进行调节
        if self.min_confidence:
            # 筛选出大于置信度阈值的预测框
            conf_keep = tf.where(class_scores >= self.min_confidence)[:, 0]
            # 取keep和conf_keep两者的交集
            # 返回结果是一个稀疏Tensor
            # sparse_indices：稀疏矩阵中那些个别元素对应的索引值
            # output_shape：输出的稀疏矩阵的shape
            # sparse_values：个别元素的值
            keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
            # 把稀疏Tensor变成正常的稠密Tensor
            keep = tf.sparse.to_dense(keep)[0]
            
        # Apply per-class NMS
        # 1. Prepare variables
        # 取出筛选后的预测框类别
        pre_nms_class_ids = tf.gather(class_ids, keep)
        # 取出筛选后的预测框概率值
        pre_nms_scores = tf.gather(class_scores, keep)
        # 取出筛选后的预测框坐标值
        pre_nms_rois = tf.gather(refined_rois,   keep)
        # 统计有哪几个不同的类别
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]
        # 非极大值抑制
        def nms_keep_map(class_id):
            # 筛选出等于当前类别的预测框的索引
            ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
            # self.max_instances默认值100，表示非极大值抑制计算后最多保留100个结果
            # self.nms_threshold默认值0.5
            class_keep = tf.image.non_max_suppression(
                    tf.gather(pre_nms_rois, ixs),
                    tf.gather(pre_nms_scores, ixs),
                    max_output_size=self.max_instances,
                    iou_threshold=self.nms_threshold)
            # 得到非极大值抑制后剩下的预测框的索引
            class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
            return class_keep

        nms_keep = []
        # 循环每一个类别，对每个类别分别做非极大值抑制
        for i in range(unique_pre_nms_class_ids.shape[0]):
            nms_keep.append(nms_keep_map(unique_pre_nms_class_ids[i]))
        # 正常来说nms_keep肯定是有值的
        if len(nms_keep) != 0:
            # 数据拼接
            nms_keep = tf.concat(nms_keep, axis=0)
        else:
            nms_keep = tf.zeros([0,], tf.int64)
        # 因为nms_keep是根据每个类别进行非极大值抑制后的结果，所以预测框的顺序是乱的
        # 根据keep中的顺序求keep和nms_keep的并集
        # 例如keep=array([0,2,4,5,8,11,15,24...203])
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
        # 再变为稠密Tensor，
        keep = tf.sparse.to_dense(keep)[0]
        # Keep top detections
        # self.max_instances默认值为100
        roi_count = self.max_instances
        # 取出nms后的预测框概率值
        class_scores_keep = tf.gather(class_scores, keep)
        # 如果nms后预测框小于100个，则使用预测框的数量作为num_keep
        # 如果nms后预测框大于100个，则只保留100
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        # 取概率最大的前num_keep个预测框
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        keep = tf.gather(keep, top_ids)  
        # 拼接数据
        # tf.gather(refined_rois, keep)概率最大的前num_keep个预测框的坐标值
        # tf.gather(class_ids, keep)概率最大的前num_keep个预测框的类别
        # tf.gather(class_scores, keep)概率最大的前num_keep个预测框的概率值
        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]
            ], axis=1)
        # detections的shape可能为(100,6)
        # 100表示100个预测框，6表示4个坐标值，1个类别值，1个概率值
        return detections
        