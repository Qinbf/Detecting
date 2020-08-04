import tensorflow as tf
import numpy as np
from detecting.models.backbones import *
from detecting.models.rpn_heads import rpn_head
from detecting.models.bbox_heads import bbox_head
from detecting.models.roi_extractors import roi_align
from detecting.models.bbox_heads import bbox_target
from detecting.datasets import transforms, utils
from detecting.utils.transforms import bbox_mapping_back


# 继承了多个类
class FasterRCNN(tf.keras.Model):
    def __init__(self, cfg, num_classes, **kwags):
        super(FasterRCNN, self).__init__(**kwags)
        # 配置
        self.cfg = cfg
        # 预测种类数
        self.num_classes = num_classes
        # ImageTransform用于处理图片
        self.img_transform = transforms.ImageTransform(cfg.DATASETS.SCALE,
                                                        cfg.DATASETS.IMG_MEAN,
                                                        cfg.DATASETS.IMG_STD,
                                                        cfg.DATASETS.PAD_MODE,
                                                        cfg.DATASETS.KEEP_ASPECT)

        # 用于得到正负样本候选区域rois和rcnn分类，回归的标签数据
        self.bbox_target = bbox_target.ProposalTarget(
            target_means=cfg.MODEL.RCNN_TARGET_MEANS,
            target_stds=cfg.MODEL.RPN_TARGET_STDS, 
            num_rcnn_deltas=cfg.MODEL.RCNN_BATCH_SIZE,
            positive_fraction=cfg.MODEL.RCNN_POS_FRAC,
            pos_iou_thr=cfg.MODEL.RCNN_POS_IOU_THR,
            neg_iou_thr_high=cfg.MODEL.RCNN_NEG_IOU_THR_HIGH,
            neg_iou_thr_low=cfg.MODEL.RCNN_NEG_IOU_THR_LOW,
            num_classes=self.num_classes)

        # 载入backbone和head_to_tail
        self.backbone, self.head_to_tail = get_backbone(cfg)

        # RPN层
        self.rpn_head = rpn_head.RPNHead(
            anchor_scales=cfg.MODEL.ANCHOR_SCALES,
            anchor_ratios=cfg.MODEL.ANCHOR_RATIOS,
            anchor_feature_strides=cfg.MODEL.ANCHOR_FEATURE_STRIDES,
            proposal_nms_top_n=cfg.MODEL.RPN_PROPOSAL_NMS_TOP_N,
            proposal_count=cfg.MODEL.RPN_PROPOSAL_COUNT,
            nms_threshold=cfg.MODEL.RPN_NMS_THRESHOLD,
            target_means=cfg.MODEL.RPN_TARGET_MEANS,
            target_stds=cfg.MODEL.RPN_TARGET_STDS,
            num_rpn_deltas=cfg.MODEL.RPN_BATCH_SIZE,
            positive_fraction=cfg.MODEL.RPN_POS_FRAC,
            pos_iou_thr=cfg.MODEL.RPN_POS_IOU_THR,
            neg_iou_thr=cfg.MODEL.RPN_NEG_IOU_THR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            name='rpn_head')

        # ROIAlign层
        self.roi_align = roi_align.ROIAlign(
            pool_shape=cfg.MODEL.POOL_SIZE,
            name='roi_align')   

        # RCNN层
        self.bbox_head = bbox_head.BBoxHead(
            num_classes=self.num_classes,
            pool_size=cfg.MODEL.POOL_SIZE,
            target_means=cfg.MODEL.RCNN_TARGET_MEANS,
            target_stds=cfg.MODEL.RCNN_TARGET_STDS,
            min_confidence=cfg.MODEL.RCNN_MIN_CONFIDENCE,
            nms_threshold=cfg.MODEL.RCNN_NMS_THRESHOLD,
            max_instances=cfg.MODEL.RCNN_MAX_INSTANCES,
            head_to_tail=self.head_to_tail,
            share_box_across_classes=cfg.MODEL.SHARE_BOX_ACROSS_CLASSES,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            name='b_box_head')

    # 计算一个批次数据的loss
    def _compute_losses(self, inputs, training=True):
        imgs, img_metas, gt_boxes, gt_class_ids = inputs
        C = self.backbone(imgs, training=training)  
        rpn_feature_maps = [C]
        rcnn_feature_maps = [C]
        # 计算rpn层的分类，回归预测
        # 得到rpn的分类预测值(1, 49152, 2)
        # 得到rpn分类概率值(1, 49152, 2)
        # 得到rpn回归预测值(1, 49152, 4)
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(
            rpn_feature_maps, training=training)
        # 根据anchors和rpn的回归值计算候选框proposals
        proposals,anchors,valid_flags = self.rpn_head.get_proposals(
            rpn_probs, rpn_deltas, img_metas, rpn_feature_maps, training)
        # 得到正负样本候选区域rois和rcnn分类，回归的标签数据
        rois, rcnn_labels, rcnn_label_weights, rcnn_delta_targets, rcnn_delta_weights = \
            self.bbox_target.build_targets(proposals, gt_boxes, gt_class_ids, img_metas)
        # 得到候选区域rois对应的特征图经过ROIAlign计算后的结果
        pooled_regions = self.roi_align((rois, rcnn_feature_maps, img_metas))

        # 对ROIAlign计算后的候选框特征图数据进行分类回归预测
        rcnn_class_logits, rcnn_probs, rcnn_deltas = \
            self.bbox_head(pooled_regions, training=training)

        # 得到rpn层分类loss和回归loss
        rpn_class_loss, rpn_bbox_loss = self.rpn_head.loss(
            rpn_class_logits, rpn_deltas, 
            gt_boxes, gt_class_ids, img_metas, anchors, valid_flags, self.cfg.SOLVER.BATCH_SIZE)
        # 得到rcnn分类loss和回归loss
        rcnn_class_loss, rcnn_bbox_loss = self.bbox_head.loss(
            rcnn_class_logits, rcnn_deltas, 
            rcnn_labels, rcnn_label_weights, rcnn_delta_targets, rcnn_delta_weights, self.cfg.SOLVER.BATCH_SIZE)
        # 返回loss
        return [rpn_class_loss, rpn_bbox_loss, 
                rcnn_class_loss, rcnn_bbox_loss]

    # 传入单张图片(3维)可以不用传img_metas
    # 传入一个批次的图片(4维)需要传img_metas
    # box_mapping_back预测的bbox结果是否映射到原始图像大小
    def predict(self, imgs, img_metas=None, box_mapping_back=True, training=False):
        if imgs.ndim==3:
            # 原始图片shape
            ori_shape = imgs.shape
            # 图片不翻转
            flip = False
            # 得到填充后的图片，resize后图片shape，缩放因子
            imgs, img_shape, scale_factor = self.img_transform(imgs, flip)
            # 填充后的图片shape
            pad_shape = imgs.shape
            img_meta_dict = dict({
            'ori_shape': ori_shape,
            'img_shape': img_shape,
            'pad_shape': pad_shape,
            'scale_factor': scale_factor,
            'flip': flip
            })
            # 把img_meta_dict中的数据组成1维的array
            img_meta = utils.compose_image_meta(img_meta_dict)
            # 图片元数据增加一个维度变成2维
            img_metas = np.expand_dims(img_meta, 0)
            # 图片数据增加一个维度变成4维
            imgs = np.expand_dims(imgs, 0)
        elif imgs.ndim==4:
            if img_metas is None:
                raise AssertionError("img_metas can't be None")
        else:
            raise AssertionError("imgs.ndim must be 3 or 4")

        C = self.backbone(imgs, training=training)  
        rpn_feature_maps = [C]
        rcnn_feature_maps = [C]
        # 计算rpn层的分类，回归预测
        # 得到rpn的分类预测值(1, 49152, 2)
        # 得到rpn分类概率值(1, 49152, 2)
        # 得到rpn回归预测值(1, 49152, 4)
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(
            rpn_feature_maps, training=training)
        # 根据anchors和rpn的回归值计算候选框proposals
        proposals,anchors,valid_flags = self.rpn_head.get_proposals(
            rpn_probs, rpn_deltas, img_metas, rpn_feature_maps, training)
        # proposals作为rois
        rois = proposals
        # 得到候选区域rois对应的特征图经过ROIAlign计算后的结果
        pooled_regions = self.roi_align((rois, rcnn_feature_maps, img_metas))

        # 对ROIAlign计算后的候选框特征图数据进行分类回归预测
        rcnn_class_logits, rcnn_probs, rcnn_deltas = \
            self.bbox_head(pooled_regions, training=training)

        # 得到rcnn最终预测结果
        detections_list = self.bbox_head.get_bboxes(
            rcnn_probs, rcnn_deltas, rois, img_metas)
        # 返回预测结果
        return self._unmold_detections(detections_list, img_metas, box_mapping_back)

    # 对一个批次图片预测结果进行处理
    def _unmold_detections(self, detections_list, img_metas, box_mapping_back):
        return [
            self._unmold_single_detection(detections_list[i], img_metas[i], box_mapping_back)
            for i in range(img_metas.shape[0])
        ]
    # 对一张图片预测结果进行处理
    def _unmold_single_detection(self, detections, img_meta, box_mapping_back):
        # 取出预测框类别不等于0的结果
        zero_ix = tf.where(tf.not_equal(detections[:, 4], 0))
        detections = tf.gather_nd(detections, zero_ix)
        # 预测框坐标
        boxes = detections[:, :4]
        # 预测框类别
        class_ids = tf.cast(detections[:, 4], tf.int32)
        # 预测框概率
        scores = detections[:, 5]
        if box_mapping_back==True:
            # 把预测框映射到原始图片大小
            boxes = bbox_mapping_back(boxes, img_meta)
        else:
            # 不映射到原始图片大小
            boxes = boxes.numpy()
        # 返回预测框数据，分类类别，该类别的预测概率
        return {'rois': boxes,
                'class_ids': class_ids.numpy(),
                'scores': scores.numpy()}