from tqdm import tqdm
import json
from pycocotools.cocoeval import COCOeval

# 评估coco数据集，批次大小为1
def eval_coco(model, generator):
    dataset_results = []
    imgIds = []
    # 循环测试数据
    for i in tqdm(range(generator.size())):
        # 生成数据
        batch_imgs, batch_metas, _, _ = generator[i]
        # 预测得到结果
        preds = model.predict(batch_imgs, batch_metas)[0]
        # 该测试图片的id
        image_id = generator.dataset.img_ids[i]
        # 保存图片id
        imgIds.append(image_id)
        # 循环该测试图片的预测结果
        for pos in range(preds['class_ids'].shape[0]):
            results = dict()
            # 置信度
            results['score'] = float(preds['scores'][pos])
            # 类别id
            results['category_id'] = int(preds['class_ids'][pos])
            # 预测框坐标
            y1, x1, y2, x2 = [float(num) for num in list(preds['rois'][pos])]
            results['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
            # 图片id
            results['image_id'] = image_id
            # 保存
            dataset_results.append(results)
    # 结果存入json文件
    with open('coco_val2017_detection_result.json', 'w') as f:
        f.write(json.dumps(dataset_results))

    # 读取json文件
    coco_dt = generator.dataset.coco.loadRes('coco_val2017_detection_result.json')
    # 结果评估
    # 修复bug，需要把pycocotools/cocoeval.py中的linspace函数的第3个参数强制转为int类型
    cocoEval = COCOeval(generator.dataset.coco, coco_dt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()