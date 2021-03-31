from detecting.utils import iou
import numpy as np
from tqdm import tqdm

# 评估模型
def evaluate(
    model,
    generator,
    iou_threshold=0.5,
):
    """ Evaluate a given dataset using a given model.
    code originally from https://github.com/fizyr/keras-retinanet
    # Arguments
        model           : The model to evaluate.
        generator       : The generator that represents the dataset to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # 类别
    classes = generator.get_categories()
    # mAP 用来保存类别对应的AP值
    average_precisions = {}
    # 记录每个类别的标准框数量
    classes_num_annotations = {}
    # 得到批次大小
    batch_size = generator.batch_size
    # 得到所有图片数量
    all_image_num = len(generator)*batch_size
    # 得到空的detections and annotations
    all_detections     = [[None for i in range(generator.num_classes())] for j in range(all_image_num)]
    all_annotations    = [[None for i in range(generator.num_classes())] for j in range(all_image_num)]
    all_scores         = [[None for i in range(generator.num_classes())] for j in range(all_image_num)]
    #  循环每张图片
    for i in tqdm(range(len(generator))):
        batch_imgs, batch_metas, batch_bboxes, batch_labels = generator[i]
        preds = model.predict(batch_imgs, batch_metas, box_mapping_back=False)

        # 一个批次可能有多张图片
        for j,pred in enumerate(preds):
            # 取出不为0的标签位置
            idx = np.where(batch_labels[j]!=0)
            # 取出不为0的真实标签
            gt_boxes = batch_bboxes[j,idx]
            # 取出不为0的真实标注框
            gt_labels = batch_labels[j,idx]
            # 预测结果不是空值
            if len(pred['class_ids'])!=0:
                # 预测概率
                scores = pred['scores']
                # 预测类别
                pred_labels = pred['class_ids']
                # 预测框
                pred_boxes = pred['rois']
                # 循环每个类别
                for label in range(generator.num_classes()):
                    # 保存每张图片的检测框预测结果
                    all_detections[i*batch_size+j][label] = pred_boxes[pred_labels == label, :]
                    # 保存每张图片的真实标注框坐标
                    all_annotations[i*batch_size+j][label] = gt_boxes[gt_labels == label, :]
                    # 保存每张图片的预测框概率值
                    all_scores[i*batch_size+j][label] = scores[pred_labels == label]    
            else:
                # 循环每个类别
                for label in range(generator.num_classes()):
                    # 保存每张图片的检测框预测结果
                    all_detections[i*batch_size+j][label] = None
                    # 保存每张图片的真实标注框坐标
                    all_annotations[i*batch_size+j][label] = gt_boxes[gt_labels == label, :]
                    # 保存每张图片的预测框概率值
                    all_scores[i*batch_size+j][label] = 0 

    # 循环每个类别
    for label in range(generator.num_classes()):
        # 假正例
        false_positives = np.zeros((0,))
        # 真正例
        true_positives  = np.zeros((0,))
        # 保存概率值
        scores          = np.zeros((0,))
        # 真实标注框数量
        num_annotations = 0.0
        # 循环所有图片
        for i in range(all_image_num):
            # 预测框
            detections           = all_detections[i][label]
            # 真实标注框
            annotations          = all_annotations[i][label]
            # 真实标注框数量
            num_annotations     += annotations.shape[0]
            # 用来保存检测到的真实标注框索引
            detected_annotations = []
            # 循环预测框
            for j,d in enumerate(detections):
                if d is not None:
                    # 保存改预测框的概率值
                    scores = np.append(scores, all_scores[i][label][j])
                    # 如果该类别真实没有真实标注框
                    if annotations.shape[0] == 0:
                        # 假正例1个
                        false_positives = np.append(false_positives, 1)
                        # 真正例0个
                        true_positives  = np.append(true_positives, 0)
                        continue
                    # 计算预测框与真实标注框交并比
                    overlaps = iou.compute_overlaps(np.expand_dims(d, axis=0), annotations)
                    # 变成numpy数据
                    overlaps = overlaps.numpy()
                    # 求预测框最大交并比对应的真实标注的索引
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    # 得到预测框与真实标注框的最大交并比
                    max_overlap = overlaps[0, assigned_annotation]
                    # 如果iou大于阈值，并且改索引不在记录索引的list中
                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        # 假正例0个
                        false_positives = np.append(false_positives, 0)
                        # 真正例1个
                        true_positives  = np.append(true_positives, 1)
                        # 把该真实标注框的索引加入list中
                        detected_annotations.append(assigned_annotation)
                    else:
                        # 假正例1个
                        false_positives = np.append(false_positives, 1)
                        # 真正例0个
                        true_positives  = np.append(true_positives, 0)
        # 关于该类别的假正例和真正例都统计完成后
        # 如果真实标注框的数量为0，那么该类别的AP等于0，可能是有bug
        if num_annotations == 0:
            average_precisions[classes[label]] = 0
            # 存入字典
            classes_num_annotations[classes[label]] = 0
            continue

        # 对预测框分数从大到小进行排序
        indices = np.argsort(-scores)
        # 根据新的索引取出假正例和真正例
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # cumsum逐次累加
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # 计算召回率，召回率是越来越高的
        recall    = true_positives / num_annotations
        # np.finfo(np.float64).eps，2.22e-16防止分母为0
        # 计算精确率，精确率是上下起伏的
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # 计算AP
        average_precision  = compute_ap(recall, precision)
        # 存入字典
        average_precisions[classes[label]] = average_precision 
        # 存入字典
        classes_num_annotations[classes[label]] = num_annotations

    return average_precisions,classes_num_annotations

# 计算AP
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # recall和precision两边填两个值
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # 精确率的值从后往前循环
    # 循环下来除了最开始的值以外，后面的值都是从高到低的形成阶梯下降
    for i in range(mpre.size - 1, 0, -1):
        # 留下大的值
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 找到recall的变化点
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # 召回率的变化乘以精确率的值
    # (mrec[i + 1] - mrec[i]) * mpre[i + 1]一段面积
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap     

# 评估并打印结果
def eval_show(model, generator, iou_threshold=0.5):
    # 获得每个类别ap字典，每个类别标注数量字典
    average_precisions,classes_num_annotations = evaluate(model,generator,iou_threshold)
    # 打印每个类别的ap，以及标签数量
    for label, ap in average_precisions.items():
        print(label+':{:.4f}'.format(ap),'  num:',classes_num_annotations[label])
    # 计算权值平均
    total_num = 0
    classes_ap = []
    for label, ap in average_precisions.items():
        # 数量
        num = classes_num_annotations[label]
        total_num += num
        classes_ap.append(ap*num)
    print('mAP:{:.4f}'.format(sum(classes_ap) / total_num))
