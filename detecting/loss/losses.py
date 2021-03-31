import tensorflow as tf
layers = tf.keras.layers
losses = tf.keras.losses

# 定义SmoothL1Loss
class SmoothL1Loss(layers.Layer):
    def __init__(self, rho=1):
        super(SmoothL1Loss, self).__init__()
        # SmoothL1Loss系数
        self._rho = rho
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        # SmoothL1Loss计算
        loss = tf.abs(y_true - y_pred)
        loss = tf.where(loss > self._rho, loss - 0.5 * self._rho, 
                        (0.5 / self._rho) * tf.square(loss))
        # 如果有sample_weight的话
        if sample_weight is not None:
            # 乘以权值
            loss = tf.multiply(loss, sample_weight)
        return loss

# rpn层分类loss
class RPNClassLoss(layers.Layer):
    def __init__(self):
        super(RPNClassLoss, self).__init__()
        # losses.Reduction.NONE返回每个样本的损失
        # losses.Reduction.SUM返回所有样本的累加损失
        # losses.Reduction.SUM_OVER_BATCH_SIZE时，返回平均损失。
        self.sparse_categorical_crossentropy = \
            losses.SparseCategoricalCrossentropy(from_logits=True,
                                                 reduction=losses.Reduction.SUM)

    def __call__(self, rpn_labels, rpn_class_logits, rpn_label_weights, batch_size):       
        # 筛选出标签不等于-1的索引
        indices = tf.where(tf.not_equal(rpn_labels, -1))
        # 取出正负样本标签数据
        rpn_labels = tf.gather_nd(rpn_labels, indices)
        # 取出正负样本标签权值
        rpn_label_weights = tf.gather_nd(rpn_label_weights, indices)
        # 取出正负样本分类预测值
        rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
        # 定义交叉熵loss计算
        loss = self.sparse_categorical_crossentropy(y_true=rpn_labels,
                                                    y_pred=rpn_class_logits,
                                                    sample_weight=rpn_label_weights)
        # loss平均
        loss = loss/batch_size
        # 返回loss
        return loss
    
# rpn层回归loss
class RPNBBoxLoss(layers.Layer):
    def __init__(self):
        super(RPNBBoxLoss, self).__init__()
        # l1平滑损失
        self.smooth_l1_loss = SmoothL1Loss()
        
    def __call__(self, rpn_delta_targets, rpn_deltas, rpn_delta_weights, batch_size):
        # 计算l1平滑损失
        loss = self.smooth_l1_loss(y_true=rpn_delta_targets, 
                                   y_pred=rpn_deltas, 
                                   sample_weight=rpn_delta_weights)
        # loss求和
        loss = tf.reduce_sum(loss)
        # loss平均
        loss = loss/batch_size
        # 返回loss
        return loss


# RCNN分类loss
class RCNNClassLoss(layers.Layer):
    def __init__(self):
        super(RCNNClassLoss, self).__init__()
        # 交叉熵
        self.sparse_categorical_crossentropy = \
            losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                 reduction=losses.Reduction.SUM)

    def __call__(self, rcnn_labels, rcnn_class_logits, rcnn_label_weights, batch_size):
        # 筛选出标签不等于-1的索引
        indices = tf.where(tf.not_equal(rcnn_labels, -1))
        # 取出正负样本标签数据
        rcnn_labels = tf.gather_nd(rcnn_labels, indices)
        # 取出正负样本标签权值
        rcnn_label_weights = tf.gather_nd(rcnn_label_weights, indices)
        # 取出正负样本分类预测值
        rcnn_class_logits = tf.gather_nd(rcnn_class_logits, indices)
        # 定义交叉熵loss计算
        loss = self.sparse_categorical_crossentropy(y_true=rcnn_labels,
                                                    y_pred=rcnn_class_logits,
                                                    sample_weight=rcnn_label_weights)
        # loss平均
        loss = loss/batch_size
        # 返回
        return loss
    
# RCNN回归loss
class RCNNBBoxLoss(layers.Layer):
    def __init__(self):
        super(RCNNBBoxLoss, self).__init__()
        # l1平滑损失
        self.smooth_l1_loss = SmoothL1Loss()
        
    def __call__(self, rcnn_delta_targets, rcnn_deltas, rcnn_delta_weights, batch_size):
        loss = self.smooth_l1_loss(y_true=rcnn_delta_targets, 
                                   y_pred=rcnn_deltas, 
                                   sample_weight=rcnn_delta_weights)
        # loss求和
        loss = tf.reduce_sum(loss)
        # loss平均
        loss = loss/batch_size
        # 返回
        return loss
