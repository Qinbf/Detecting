import tensorflow as tf

from detecting.utils import transforms
from detecting.utils import iou
from detecting.utils.misc import trim_zeros
# anchor标签
class AnchorTarget(object):
    def __init__(self,
                 target_means=(0., 0., 0., 0.), 
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3):
        '''Compute regression and classification targets for anchors.
        
        Attributes
        ---
            target_means: [4]. Bounding box refinement mean for RPN.
            target_stds: [4]. Bounding box refinement standard deviation for RPN.
            num_rpn_deltas: int. Maximal number of Anchors per image to feed to rpn heads.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rpn_deltas = num_rpn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
    # 得到分类标签，分类权重，回归标签，回归权重
    def build_targets(self, anchors, valid_flags, gt_boxes, gt_class_ids):
        '''Given the anchors and GT boxes, compute overlaps and identify positive
        anchors and deltas to refine them to match their corresponding GT boxes.
        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image 
                coordinates.
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.
        ---
            anchors: 多尺度的anchors
            valid_flags: anchors是否有效的标志
            gt_boxes: 真实标注框
            gt_class_ids: 真实标注框类别标签
        Returns
        ---
            rpn_labels: [batch_size, num_anchors] 
                Matches between anchors and GT boxes. 1 - positive samples; 0 - negative samples; -1 - neglect
            rpn_label_weights: [batch_size, num_anchors] 
            rpn_delta_targets: [batch_size, num_anchors, (dy, dx, log(dh), log(dw))] 
                Anchor bbox deltas.
            rpn_delta_weights: [batch_size, num_anchors, 4]
        '''
        # 用于保存数据
        rpn_labels = []
        rpn_label_weights = []
        rpn_delta_targets = []
        rpn_delta_weights = []
        # 一个批次图片数量
        num_imgs = gt_class_ids.shape[0]
        # 循环每张图片
        for i in range(num_imgs):
            # 得到分类标签，分类权重，回归标签，回归权重
            labels, label_weights, delta_targets, delta_weights = self._build_single_target(
                anchors, valid_flags[i], gt_boxes[i], gt_class_ids[i])
            rpn_labels.append(labels)
            rpn_label_weights.append(label_weights)
            rpn_delta_targets.append(delta_targets)
            rpn_delta_weights.append(delta_weights)

        # 对一个批次保存的数据进行堆叠
        rpn_labels = tf.stack(rpn_labels)
        rpn_label_weights = tf.stack(rpn_label_weights)
        rpn_delta_targets = tf.stack(rpn_delta_targets)
        rpn_delta_weights = tf.stack(rpn_delta_weights)
        # 返回数据
        return rpn_labels, rpn_label_weights, rpn_delta_targets, rpn_delta_weights

    # 得到分类标签，分类权重，回归标签，回归权重
    def _build_single_target(self, anchors, valid_flags, gt_boxes, gt_class_ids):
        '''Compute targets per instance.
        
        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)]
            valid_flags: [num_anchors]
            gt_class_ids: [num_gt_boxes]
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
        ---
            anchors: anchors
            valid_flags: anchors是否有效的标志
            gt_class_ids: 真实标注框类别标签
            gt_boxes: 真实标注框
            操作流程：
            1.筛选出有效的anchors
            2.生成跟anchors数量相同的-1标签，-1表示中立样本
            3.计算anchors和真实标注框的交并比对比结果
            4.根据交并比结果筛选正样本和负样本，IOU>0.7正样本，<0.3负样本，其余中立样本
            5.去掉多余的正样本和负样本，正样本不超过128个，正负样本一共256个
            6.计算anchors与真实标注框的(dy,dx,dh,dw)误差
            7.分配分类任务中和回归任务中的权重，分类任务训练正负样本，回归任务只训练正样本
            8.把数据长度都填充到原始anchors的长度
        Returns
        ---
            labels: [num_anchors]
            label_weights: [num_anchors]
            delta_targets: [num_anchors, (dy, dx, log(dh), log(dw))] 
            delta_weights: [num_anchors, 4]
        '''

        # 得到数值不为0的真实标注框
        gt_boxes, _ = trim_zeros(gt_boxes)
        # anchors数量，比如一共有49152个anchors
        total_anchors = anchors.shape[0]    
        # 选出有效的anchors
        valid_flags_ids = tf.where(tf.equal(valid_flags, 1))
        valid_flags = tf.gather_nd(valid_flags, valid_flags_ids)
        anchors = tf.gather_nd(anchors, valid_flags_ids)

        # 生成跟anchors数量相同的-1标签
        labels = -tf.ones(anchors.shape[0], dtype=tf.int32)
        
        # 计算anchors和真实标注框的交并比对比结果
        # 例如overlaps的shape可能为(29420,9)，29420是anchors数量，9是真实标注框数量
        overlaps = iou.compute_overlaps(anchors, gt_boxes)

        # 取29420个候选框与真实标注框的最大IOU所在索引
        anchor_iou_argmax = tf.argmax(overlaps, axis=1)


        # 候选框与真实标注框的最大IOU
        anchor_iou_max = tf.reduce_max(overlaps, axis=[1])
        # self.neg_iou_thr默认值为0.3
        # 小于IOU阈值的anchors标注为0
        labels = tf.where(anchor_iou_max < self.neg_iou_thr, 
                          tf.zeros(anchors.shape[0], dtype=tf.int32), labels)

        # self.pos_iou_thr默认值为0.7
        # 大于IOU阈值的anchors标注为1
        labels = tf.where(anchor_iou_max >= self.pos_iou_thr, 
                          tf.ones(anchors.shape[0], dtype=tf.int32), labels)

        # 为了防止所有anchors与GT的IOU都小于阈值0.7，下面我们给每个GT都找一个IOU最大的anchor
        # 计算每个真实标注框对应的IOU最大的anchor
        # 比如有6个真实标注框，这里就得到6个anchor的索引
        gt_iou_argmax = tf.argmax(overlaps, axis=0)
        # 把这6个anchor在labels中的对应的值设置为1
        labels = tf.tensor_scatter_nd_update(labels, 
                                             tf.reshape(gt_iou_argmax, (-1, 1)), 
                                             tf.ones(gt_iou_argmax.shape, dtype=tf.int32))
        

        # 得到正样本的索引
        ids = tf.where(tf.equal(labels, 1))

        # 不要让正样本的anchor超过一半
        # 得到正样本的索引
        # self.num_rpn_deltas默认值256，self.positive_fraction默认值0.5
        # 正样本的数量128
        extra = ids.shape.as_list()[0] - int(self.num_rpn_deltas * self.positive_fraction)
        # extra>0表示多余的正样本数量
        if extra > 0:
            # 随机选择多余的正样本
            ids = tf.random.shuffle(ids)[:extra]
            # 把多余的正样本标签更新为-1，变成中立样本
            labels = tf.tensor_scatter_nd_update(labels, 
                                                 ids, 
                                                 -tf.ones(ids.shape[0], dtype=tf.int32))
        # 得到负样本的索引
        ids = tf.where(tf.equal(labels, 0))
        # self.num_rpn_deltas默认值为256
        # 负样本数量-(256-正样本数量)
        extra = ids.shape.as_list()[0] - (self.num_rpn_deltas -
            tf.reduce_sum(tf.cast(tf.equal(labels, 1), tf.int32)))
        # extra>0表示多余的负样本数量
        if extra > 0:
            # 随机选择多余的负样本
            ids = tf.random.shuffle(ids)[:extra]
            # 把多余的正样本标签更新为-1，变成中立样本
            labels = tf.tensor_scatter_nd_update(labels, 
                                                 ids, 
                                                 -tf.ones(ids.shape[0], dtype=tf.int32))


        # 取出每个anchors对应IOU最大的真实标注框
        gt = tf.gather(gt_boxes, anchor_iou_argmax)
        # 计算anchors与真实标注框的(dy,dx,dh,dw)误差
        delta_targets = transforms.bbox2delta(anchors, gt, self.target_means, self.target_stds)

        # 计算分类任务中的权重
        label_weights = tf.zeros((anchors.shape[0],), dtype=tf.float32)
        # 计算回归任务中的权重
        delta_weights = tf.zeros((anchors.shape[0],), dtype=tf.float32)
        # 得到正负样本的数量
        num_bfg = tf.where(tf.greater_equal(labels, 0)).shape[0]
        # num_bfg正常情况下应该都是大于0
        if num_bfg > 0:
            # 正负样本分类都是相同的权重
            label_weights = tf.where(labels >= 0, 
                                     tf.ones(label_weights.shape, dtype=tf.float32)/ num_bfg, 
                                     label_weights)
            # 回归只有正样本才有权重
            delta_weights = tf.where(labels > 0, 
                                    tf.ones(delta_weights.shape, dtype=tf.float32) / num_bfg, 
                                     delta_weights)
        else:
            # 如果num_bfg为0发出警告
            warnings.warn("num_bfg==0", RuntimeWarning)

        # reshape后复制为4列，shape为(29420, 4)
        delta_weights = tf.tile(tf.reshape(delta_weights, (-1, 1)), [1, 4])
        # 把数据长度都填充为total_anchors
        labels = self._unmap(labels, total_anchors, valid_flags_ids, -1)
        label_weights = self._unmap(label_weights, total_anchors, valid_flags_ids, 0)
        delta_targets = self._unmap(delta_targets, total_anchors, valid_flags_ids, 0)
        delta_weights = self._unmap(delta_weights, total_anchors, valid_flags_ids, 0)


        # 分类标签，分类权重，回归标签，回归权重
        return labels, label_weights, delta_targets, delta_weights

    # 用于数据填充
    def _unmap(self, data, count, inds, fill=0):
        if len(data.shape) == 1:
            ret = tf.fill((count,), fill)
            ret = tf.cast(ret,dtype=data.dtype)
            ret = tf.tensor_scatter_nd_update(ret, 
                                              inds, 
                                              data)
        else:
            ret = tf.fill((count,)+data.shape[1:], fill)
            ret = tf.cast(ret,dtype=data.dtype)
            ret = tf.tensor_scatter_nd_update(ret, 
                                              inds, 
                                              data)
        return ret