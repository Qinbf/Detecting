import tensorflow as tf
layers = tf.keras.layers

from detecting.utils import transforms
from detecting.utils.misc import *

from detecting.models.anchors import anchor_generator, anchor_target
from detecting.loss import losses

# RPN层
class RPNHead(tf.keras.Model):
    def __init__(self, 
                 anchor_scales=(32, 64, 128, 256, 512), 
                 anchor_ratios=(0.5, 1, 2), 
                 anchor_feature_strides=(4, 8, 16, 32, 64),
                 proposal_nms_top_n=12000,
                 proposal_count=2000, 
                 nms_threshold=0.7, 
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2), 
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3,
                 weight_decay=0,
                 if_fpn=True,
                 **kwags):
        '''Network head of Region Proposal Network.

                                      / - rpn_cls (1x1 conv)
        input - rpn_conv (3x3 conv) -
                                      \ - rpn_reg (1x1 conv)

        Attributes
        ---
            anchor_scales: 1D array of anchor sizes in pixels.
            anchor_ratios: 1D array of anchor ratios of width/height.
            anchor_feature_strides: Stride of the feature map relative 
                to the image in pixels.
            proposal_nms_top_n: int.RPN proposals kept before non-maximum 
                supression.
            proposal_count: int. RPN proposals kept after non-maximum 
                supression.
            nms_threshold: float. Non-maximum suppression threshold to 
                filter RPN proposals.
            target_means: [4] Bounding box refinement mean.
            target_stds: [4] Bounding box refinement standard deviation.
            num_rpn_deltas: int.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''
        super(RPNHead, self).__init__(**kwags)
        self.proposal_nms_top_n = proposal_nms_top_n
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.target_means = target_means
        self.target_stds = target_stds
        self.weight_decay = weight_decay
        self.if_fpn = if_fpn

        # anchor生成器
        self.generator = anchor_generator.AnchorGenerator(
            scales=anchor_scales, 
            ratios=anchor_ratios, 
            feature_strides=anchor_feature_strides)
        # anchor标签
        self.anchor_target = anchor_target.AnchorTarget(
            target_means=target_means, 
            target_stds=target_stds,
            num_rpn_deltas=num_rpn_deltas,
            positive_fraction=positive_fraction,
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr)
        # rpn层分类loss，交叉熵
        self.rpn_class_loss = losses.RPNClassLoss()
        # rpn层回归loss
        self.rpn_bbox_loss = losses.RPNBBoxLoss()

        # 初始化
        init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)
        # 正则化
        reg = tf.keras.regularizers.l2(self.weight_decay)

        # 3*3卷积
        self.rpn_conv_shared = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer=init, kernel_regularizer=reg,
                                             name='rpn_conv_shared')

        # 预测所有anchors的分类结果
        self.rpn_class_raw = layers.Conv2D(2 * len(anchor_ratios) * len(anchor_scales), (1, 1), kernel_initializer=init, kernel_regularizer=reg,
                                        name='rpn_class_raw')
        # 预测所有anchors的回归结果
        self.rpn_delta_pred = layers.Conv2D(4 * len(anchor_ratios)* len(anchor_scales), (1, 1), kernel_initializer=init, kernel_regularizer=reg,
                                        name='rpn_bbox_pred')

        
    # 计算rpn层的分类，回归预测
    def __call__(self, inputs, training=True):
        '''
        Args
        ---
            inputs: [batch_size, feat_map_height, feat_map_width, channels] 
                one level of pyramid feat-maps.
        
        Returns
        ---
            rpn_class_logits: [batch_size, num_anchors, 2]
            rpn_probs: [batch_size, num_anchors, 2]
            rpn_deltas: [batch_size, num_anchors, 4]
        '''
        # 3*3卷积
        shared = self.rpn_conv_shared(inputs[0])
        shared = tf.nn.relu6(shared)
        # 分类计算
        x = self.rpn_class_raw(shared)
        # 4维变3维(批次大小，anchor数量，2分类)
        rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])
        rpn_probs = tf.nn.softmax(rpn_class_logits)
        # 回归计算
        x = self.rpn_delta_pred(shared)
        # 4维变3维(批次大小，anchor数量，4个回归值)
        rpn_deltas = tf.reshape(x, [tf.shape(x)[0], -1, 4])
        # 返回
        return rpn_class_logits, rpn_probs, rpn_deltas
   
          
    # rpn层loss定义
    def loss(self, rpn_class_logits, rpn_deltas, gt_boxes, gt_class_ids, img_metas, anchors, valid_flags, batch_size):
        '''Calculate rpn loss
        '''
        # 得到分类标签，分类权重，回归标签，回归权重
        rpn_labels, rpn_label_weights, rpn_delta_targets, rpn_delta_weights = \
            self.anchor_target.build_targets(anchors, valid_flags, gt_boxes, gt_class_ids)
        # 计算rpn层分类loss，交叉熵
        rpn_class_loss = self.rpn_class_loss(
            rpn_labels, rpn_class_logits, rpn_label_weights, batch_size)
        # 计算rpn层回归loss，l1平滑损失
        rpn_bbox_loss = self.rpn_bbox_loss(
            rpn_delta_targets, rpn_deltas, rpn_delta_weights, batch_size)
        # 返回分类loss和回归loss
        return rpn_class_loss, rpn_bbox_loss
    
    # 根据anchors和rpn的回归值计算候选框proposals
    def get_proposals(self, 
                      rpn_probs, 
                      rpn_deltas, 
                      img_metas, 
                      rpn_feature_maps,
                      training,
                      with_probs=False):
        '''Calculate proposals.
        
        Args
        ---
            rpn_probs: [batch_size, num_anchors, (bg prob, fg prob)]
            rpn_deltas: [batch_size, num_anchors, (dy, dx, log(dh), log(dw))]
            img_metas: [batch_size, 12]
            rpn_feature_maps: [batch_size, feat_map_height, feat_map_width, channels] 
                                one level of pyramid feat-maps.
            with_probs: bool.
        ---
            rpn_probs: rpn层分类预测
            rpn_deltas: rpn层回归预测
            img_metas: 图像元数据
            rpn_feature_maps: 特征图不使用fpn的时候需要用到
            with_probs: 输出结果是否包含候选框分类概率值
        Returns
        ---
            proposals: [batch_size * num_proposals, (batch_ind, y1, x1, y2, x2))] in 
                normalized coordinates if with_probs is False. 
                Otherwise, the shape of proposals in proposals_list is 
                [batch_size * num_proposals, (batch_ind, y1, x1, y2, x2, probs)]
        
        '''
        #  得到特征图高度和宽度的shape
        feature_shape = rpn_feature_maps[0].get_shape().as_list()[1:-1]
        # 得到anchors和valid_flags
        anchors, valid_flags = self.generator.generate_anchors(img_metas, feature_shape)
        # 取出anchor的某个概率值，把0位置作为背景，1位置作为物体，用来表示anchor正样本的概率
        rpn_probs = rpn_probs[:, :, 1]
        # 得到填充后的图片shape
        pad_shapes = calc_pad_shapes(img_metas)
        # for i in range(img_metas.shape[0])为了循环一个批次的数据
        # 对anchors进行处理得到候选框proposals
        proposals_list = [
            self._get_proposals_single(
                rpn_probs[i], rpn_deltas[i], anchors, valid_flags[i], pad_shapes[i], i, training, with_probs)
            for i in range(img_metas.shape[0])
        ]
        # 对批次数据进行拼接
        # proposals: [batch_size * num_proposals, (batch_ind, y1, x1, y2, x2)]
        proposals = tf.concat(proposals_list, axis=0)
        # 返回anchor,valid_flags
        return proposals,anchors,valid_flags
    
    # 对anchors进行处理得到候选框proposals
    def _get_proposals_single(self, 
                              rpn_probs, 
                              rpn_deltas, 
                              anchors, 
                              valid_flags, 
                              img_shape,
                              batch_ind,
                              training,
                              with_probs):
        '''Calculate proposals.
        
        Args
        ---
            rpn_probs: [num_anchors]
            rpn_deltas: [num_anchors, (dy, dx, log(dh), log(dw))]
            anchors: [num_anchors, (y1, x1, y2, x2)] anchors defined in 
                pixel coordinates.
            valid_flags: [num_anchors]
            img_shape: np.ndarray. [2]. (img_height, img_width)
            batch_ind: int.
            training: bool.
            with_probs: bool.
        ---
            rpn_probs: rpn层分类预测
            rpn_deltas: rpn层回归预测
            anchors: 固定生成的候选框
            valid_flags: anchors是否有效的标注
            img_shape: resize后图片的高度和宽度
            training: 模型是否为训练状态
            with_probs: 输出结果是否包含候选框分类概率值
        ---
            程序步骤：
            1.选出有效的anchors(排除图片填充区域的anchors)
            2.选出概率最大的前12000个anchors
            3.使用rpn预测回归值对anchors进行调整得到proposals
            4.把proposals限制在图片范围内
            5.对proposals进行数值归一化和非极大值抑制只留下2000个候选区域
            6.再对proposals进行一些处理最后返回[num_proposals, (batch_ind, y1, x1, y2, x2)]
        Returns
        ---
            proposals: [num_proposals, (batch_ind, y1, x1, y2, x2)] in normalized 
                coordinates.
        '''
        # resize后图片的高度和宽度
        H, W = img_shape
        # resize后的图片的像素范围
        window = tf.constant([0., 0., H, W], dtype=tf.float32)

        # 训练阶段去除所有无效的anchor
        if training == True:
            # valid_flags变成布尔类型
            valid_flags = tf.cast(valid_flags, tf.bool)
            # 只保留有效anchors的正样本预测概率值rpn_probs
            rpn_probs = tf.boolean_mask(rpn_probs, valid_flags)
            # 只保留有效anchors的正样本预测回归值rpn_deltas
            rpn_deltas = tf.boolean_mask(rpn_deltas, valid_flags)
            # 只保留有效anchors
            anchors = tf.boolean_mask(anchors, valid_flags)
        # 预测阶段把所有anchors都clip到图片内
        else:
            anchors = transforms.bbox_clip(anchors, window)

        # 设置一个anchors数量的阈值
        # self.proposal_nms_top_n为12000
        pre_nms_limit = min(self.proposal_nms_top_n, anchors.shape[0])
        # 取出概率最大的前12000个预测值所在的位置
        ix = tf.nn.top_k(rpn_probs, pre_nms_limit, sorted=True).indices
        # 取出概率最大的前12000个预测值
        rpn_probs = tf.gather(rpn_probs, ix)
        # 取出概率最大的前12000个预测框回归值
        rpn_deltas = tf.gather(rpn_deltas, ix)
        # 取出概率最大的前12000个anchors坐标值
        anchors = tf.gather(anchors, ix)
        
        # 对anchors进行调整
        proposals = transforms.delta2bbox(anchors, rpn_deltas, 
                                          self.target_means, self.target_stds)
        

        # 把调整后的anchors限制在图片范围内
        proposals = transforms.bbox_clip(proposals, window)
        
        # 数值归一化
        proposals = proposals / tf.constant([H, W, H, W], dtype=tf.float32)
        
        # 非极大值抑制NMS
        # self.proposal_count:2000
        # self.nms_threshold:0.7
        indices = tf.image.non_max_suppression(
            proposals, rpn_probs, self.proposal_count, self.nms_threshold)
        # 得到进行了非极大值抑制后的概率最大的前2000个候选区域
        proposals = tf.gather(proposals, indices)
        # 如果输出要包含分类概率
        if with_probs:
            # 得到这2000个候选区域的分类概率
            proposal_probs = tf.expand_dims(tf.gather(rpn_probs, indices), axis=1)
            # 再与proposals数据拼接起来
            proposals = tf.concat([proposals, proposal_probs], axis=1)

        # 万一经过非极大值抑制后没有2000个候选区域
        # 计算一下要填充多少个数据
        padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
        # [(0, padding), (0, 0)]分别表示上下左右的填充，这里表示向下填充padding个数据，默认填充0
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        # batch_ind表示批次中的第几张图片
        batch_inds = tf.ones((proposals.shape[0], 1)) * batch_ind
        # 拼接后数据格式为[num_proposals, (batch_ind, y1, x1, y2, x2)]
        proposals = tf.concat([batch_inds, proposals], axis=1)
        # 返回数据
        return proposals
        
        