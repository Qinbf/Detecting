import numpy as np
import tensorflow as tf
import warnings
from detecting.utils import transforms
from detecting.utils import iou
from detecting.utils.misc import *
# RCNN标签
class ProposalTarget(object):
    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2), 
                 num_rcnn_deltas=256,
                 positive_fraction=0.25,
                 pos_iou_thr=0.5,
                 neg_iou_thr_high=0.5,
                 neg_iou_thr_low=0.1,
                 num_classes=81):
        '''Compute regression and classification targets for proposals.
        
        Attributes
        ---
            target_means: [4]. Bounding box refinement mean for RCNN.
            target_stds: [4]. Bounding box refinement standard deviation for RCNN.
            num_rcnn_deltas: int. Maximal number of RoIs per image to feed to bbox heads.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr_high: float.
            neg_iou_thr_low: float.
            num_classes: int.

        '''
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rcnn_deltas = num_rcnn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr_high = neg_iou_thr_high
        self.neg_iou_thr_low = neg_iou_thr_low
        self.num_classes = num_classes
            
    # 得到正负样本候选区域rois，分类和回归的标签数据
    def build_targets(self, proposals, gt_boxes, gt_class_ids, img_metas):
        '''Generates detection targets for images. Subsamples proposals and
        generates target class IDs, bounding box deltas for each.
        
        Args
        ---
            proposals: [batch_size * num_proposals, (batch_ind, y1, x1, y2, x2)] in normalized coordinates.
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image coordinates.
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.
            img_metas: [batch_size, 12]
        ---
            proposals: rpn层产生的候选框
            gt_boxes: 真实标注框
            gt_class_ids: 真实标注框的类别
            img_metas: 图片元数据
        Returns
        ---
            rcnn_rois: [batch_size * num_rois, (batch_ind, y1, x1, y2, x2)] in normalized coordinates
            rcnn_labels: [batch_size * num_rois].
                Integer class IDs.
            rcnn_label_weights: [batch_size * num_rois].
            rcnn_delta_targets: [batch_size * num_rois, num_classes, (dy, dx, log(dh), log(dw))].
                ROI bbox deltas.
            rcnn_delta_weights: [batch_size * num_rois, num_classes, 4].
            
        '''
        # 得到填充后的图片shape
        pad_shapes = calc_pad_shapes(img_metas)
        # 得到批次大小
        batch_size = img_metas.shape[0]
        # 把proposals的shape变成(batch_size,num_rois,5)
        proposals = tf.reshape(proposals[:, :5], (batch_size, -1, 5))
        # 用于保存数据
        rcnn_rois = []
        rcnn_labels = []
        rcnn_label_weights = []
        rcnn_delta_targets = []
        rcnn_delta_weights = []
        # 循环一个批次数据
        for i in range(batch_size):
            # 对候选区域和真实标注框进行处理和计算，得到分类和回归的标签数据
            rois, labels, label_weights, delta_targets, delta_weights= self._build_single_target(
                proposals[i], gt_boxes[i], gt_class_ids[i], pad_shapes[i], i)
            # rois的shape为(256,5),[num_rois, (batch_ind, y1, x1, y2, x2)]
            rcnn_rois.append(rois)
            # labels的shape为(256,)
            rcnn_labels.append(labels)
            # label_weights的shape为(256,)
            rcnn_label_weights.append(label_weights)
            # delta_targets的shape为(256, num_classes, 4)
            rcnn_delta_targets.append(delta_targets)
            # delta_weights的shape为(256, num_classes, 4)
            rcnn_delta_weights.append(delta_weights)

        # 对一个批次保存的数据进行拼接
        rcnn_rois = tf.concat(rcnn_rois, axis=0)
        rcnn_labels = tf.concat(rcnn_labels, axis=0)
        rcnn_label_weights = tf.concat(rcnn_label_weights, axis=0)
        rcnn_delta_targets = tf.concat(rcnn_delta_targets, axis=0)
        rcnn_delta_weights = tf.concat(rcnn_delta_weights, axis=0)
        # 返回数据
        return rcnn_rois, rcnn_labels, rcnn_label_weights, rcnn_delta_targets, rcnn_delta_weights
    
    # 对候选区域和真实标注框进行处理和计算，得到正负样本候选区域rois和分类，回归的标签数据
    def _build_single_target(self, proposals, gt_boxes, gt_class_ids, img_shape, batch_ind):
        '''
        Args
        ---
            proposals: [num_proposals, (batch_ind, y1, x1, y2, x2)] in normalized coordinates.
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
            gt_class_ids: [num_gt_boxes]
            img_shape: np.ndarray. [2]. (img_height, img_width)
            batch_ind: int.
        ---
            proposals: anchors调整后得到的候选框
            gt_boxes: 真实标注框
            gt_class_ids: 真实标注框对应的分类
            img_shape: 填充后的图片形状
            batch_ind: 批次中的index
            操作流程：
            1.去除之前填充的0值候选框
            2.对真实标注框的坐标数据进行归一化
            3.计算候选框和真实标注框的交并比对比结果
            4.根据交并比结果筛选正样本和负样本
            5.合并正负样本并填充到256个样本
            6.计算候选框与真实标注框的(dy,dx,dh,dw)误差
            7.分配分类任务中和回归任务中的权重，分类任务训练正负样本，回归任务只训练正样本
        Returns
        ---
            rois: [num_rois, (batch_ind, y1, x1, y2, x2)]
            labels: [num_rois]
            label_weights: [num_rois]
            target_deltas: [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
            delta_weights: [num_rois, num_classes, 4]
        '''
        # 得到填充后的图片的高度和宽度
        H, W = img_shape
        # 得到数值不为0的候选框（去除之前填充的0值候选框）
        trimmed_proposals, _ = trim_zeros(proposals[:, 1:])
        # 得到数值不为0的真实标注框
        gt_boxes, non_zeros = trim_zeros(gt_boxes)

        # 根据真实标注框的情况筛选真实标注框的标注，去除标注框为0的标注
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros)
        # 对真实标注框的坐标数据进行归一化
        gt_boxes = gt_boxes / tf.constant([H, W, H, W], dtype=tf.float32)
        # 计算候选框和真实标注框的交并比对比结果
        # 例如overlaps的shape可能为(2000,6)，2000是候选框数量，6是真实标注框数量
        overlaps = iou.compute_overlaps(trimmed_proposals, gt_boxes)
        
        # 取2000个候选框与真实标注框的最大IOU
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # self.pos_iou_thr默认值为0.5
        # 计算得到大于IOU阈值的候选框的索引
        positive_indices = tf.where(roi_iou_max >= self.pos_iou_thr)[:, 0]
        # self.neg_iou_thr_high默认值为0.5
        # self.neg_iou_thr_low默认值为0.1
        # 计算得到小于IOU上阈值大于IOU下阈值的候选框的索引
        negative_indices = tf.where((roi_iou_max < self.neg_iou_thr_high)&(roi_iou_max > self.neg_iou_thr_low))[:, 0]
        # 计算得到小于IOU下阈值的索引，用作负样本的填充
        negative_indices2 = tf.where(roi_iou_max <= self.neg_iou_thr_low)[:, 0]

        # self.num_rcnn_deltas默认值256
        # self.positive_fraction默认值0.25
        # 正样本64个
        positive_count = int(self.num_rcnn_deltas * self.positive_fraction)
        # 注意这里的positive_indices不一定有64个，也有可能会小于64个，比如53个
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        # 计算一下正样本的数量
        positive_count = tf.shape(positive_indices)[0]
        
        # 下面计算正负样本的比例为1:3
        r = 1.0 / self.positive_fraction
        # 负样本的数量，正样本53个，负样本就会有159个
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        # 有可能negative_indices不足negative_count个
        negative_count = tf.minimum(negative_count, negative_indices.shape[0])
        # 取出负样本
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

        # 计算需要填充的样本数
        P = tf.maximum(self.num_rcnn_deltas - positive_count - negative_count, 0)
        # 有可能negative_indices2不足P个
        P = tf.minimum(P, negative_indices2.shape[0])
        # 取出填充的负样本
        negative_indices2 = tf.random.shuffle(negative_indices2)[:P]
        # 合并
        negative_indices = tf.concat([negative_indices,negative_indices2], axis=0)
        
        # 得到正负样本的proposals，比如正样本数据shape为(53, 5)，负样本数据shape为(203, 5)
        positive_rois = tf.gather(proposals, positive_indices) 
        negative_rois = tf.gather(proposals, negative_indices) 

        # 取出作为正样本的proposals与真实标注框的IOU
        positive_overlaps = tf.gather(overlaps, positive_indices)
        # 53个候选框与真实标注框的最大IOU所在的索引
        roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
        # 根据索引得到53个候选框每个候选框所对应的真实标注框坐标数据,shape为(53,4)
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        # 得到53个候选框每个候选框所对应的真实标注框标签数据,shape为(53,)
        labels = tf.gather(gt_class_ids, roi_gt_box_assignment)

        
        # 计算候选框与真实标注框的(dy,dx,dh,dw)误差
        delta_targets = transforms.bbox2delta(
            positive_rois[:, 1:], roi_gt_boxes, self.target_means, self.target_stds)

        # 合并正负样本
        rois = tf.concat([positive_rois, negative_rois], axis=0)

        # 给labels填充N个0
        labels = tf.pad(labels, [(0, negative_count+P)], constant_values=0)
        # 候选框与真实标注框的(dy,dx,dh,dw)误差下方填充N+P行0，因为只对正样本进行回归，负样本不做回归
        delta_targets = tf.pad(delta_targets, [(0, negative_count+P), (0, 0)]) 

        # 正负样本的总数量
        num_bfg = rois.shape[0]
        if num_bfg!=self.num_rcnn_deltas:
            P_zero = self.num_rcnn_deltas - num_bfg
            # 给rois下方填充P行0数据
            rois = tf.pad(rois, [(0, P_zero), (0, 0)]) 
            # 给labels填充P_zero个-1,
            labels = tf.pad(labels, [(0, P_zero)], constant_values=-1)
            # 候选框与真实标注框的(dy,dx,dh,dw)误差下方填充P_zero行0，因为只对正样本进行回归，负样本不做回归
            delta_targets = tf.pad(delta_targets, [(0, P_zero), (0, 0)]) 


        # 计算分类任务中的权重
        label_weights = tf.zeros((self.num_rcnn_deltas,), dtype=tf.float32)
        # 计算回归任务中的权重
        delta_weights = tf.zeros((self.num_rcnn_deltas,), dtype=tf.float32)

        # 正样本数量大于0时才做训练
        if positive_count > 0:
            # 正负样本分类都是相同的权重
            label_weights = tf.where(labels >= 0,
                                        tf.ones((self.num_rcnn_deltas,), dtype=tf.float32)/ self.num_rcnn_deltas,
                                        label_weights)
            # 回归只有正样本才有权重
            delta_weights = tf.where(labels > 0,
                                        tf.ones((self.num_rcnn_deltas,), dtype=tf.float32) / self.num_rcnn_deltas,
                                        delta_weights)
        else:
            pass

        # reshape后复制为4列，shape为(256, 4)
        delta_weights = tf.tile(tf.reshape(delta_weights, (-1, 1)), [1, 4]) 

        # 返回正负样本候选框，分类标签，分类权重，回归标签，回归权重
        return rois, labels, label_weights, delta_targets, delta_weights
