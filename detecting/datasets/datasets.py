from detecting.datasets import coco,voc,mydata
from detecting.datasets import data_generator
import tensorflow as tf

# 载入数据
def load_data(cfg):
    # 载入COCO数据集
    if cfg.DATASETS.NAMES == 'COCO':
        dataset = coco.CocoDataSet(dataset_dir=cfg.DATASETS.ROOT_DIR,
                                        subset=cfg.DATASETS.SUBSET,
                                        flip_ratio=cfg.DATASETS.FLIP_RATIO,
                                        pad_mode=cfg.DATASETS.PAD_MODE,
                                        mean=cfg.DATASETS.IMG_MEAN,
                                        std=cfg.DATASETS.IMG_STD,
                                        scale=cfg.DATASETS.SCALE,
                                        keep_aspect=cfg.DATASETS.KEEP_ASPECT,
                                        image_dir=cfg.DATASETS.IMAGE_DIR,
                                        label_dir=cfg.DATASETS.LABEL_DIR,)
    # 载入VOC数据集
    elif cfg.DATASETS.NAMES == 'VOC':
        dataset = voc.VocDataSet(dataset_dir=cfg.DATASETS.ROOT_DIR,
                                        subset=cfg.DATASETS.SUBSET,
                                        flip_ratio=cfg.DATASETS.FLIP_RATIO,
                                        pad_mode=cfg.DATASETS.PAD_MODE,
                                        mean=cfg.DATASETS.IMG_MEAN,
                                        std=cfg.DATASETS.IMG_STD,
                                        scale=cfg.DATASETS.SCALE,
                                        keep_aspect=cfg.DATASETS.KEEP_ASPECT,
                                        image_dir=cfg.DATASETS.IMAGE_DIR,
                                        label_dir=cfg.DATASETS.LABEL_DIR,)
    # 载入自己的数据集
    elif cfg.DATASETS.NAMES == 'MYDATA':
        dataset = mydata.MyDataSet(dataset_dir=cfg.DATASETS.ROOT_DIR,
                                        subset=cfg.DATASETS.SUBSET,
                                        classes=cfg.DATASETS.CLASSES,
                                        flip_ratio=cfg.DATASETS.FLIP_RATIO,
                                        pad_mode=cfg.DATASETS.PAD_MODE,
                                        mean=cfg.DATASETS.IMG_MEAN,
                                        std=cfg.DATASETS.IMG_STD,
                                        scale=cfg.DATASETS.SCALE,
                                        keep_aspect=cfg.DATASETS.KEEP_ASPECT,
                                        image_dir=cfg.DATASETS.IMAGE_DIR,
                                        label_dir=cfg.DATASETS.LABEL_DIR,)
    else:
        raise AssertionError('datasets name must be "COCO" , "VOC" , "MYDATA" ')
    # 返回数据集
    return dataset

# 返回数据生成器
def load_data_generator(cfg, shuffle=False):
    # 载入数据
    dataset = load_data(cfg)
    # 批次大小
    batch_size = cfg.SOLVER.BATCH_SIZE
    # 定义数据生成器
    generator = data_generator.DataGenerator(dataset, batch_size=batch_size, shuffle=shuffle)
    return generator

# 返回测试数据生成器
def load_test_data_generator(cfg, shuffle=False):
    # 批次大小
    batch_size = cfg.SOLVER.BATCH_SIZE
    # 定义数据生成器
    generator = data_generator.TestDataGenerator(cfg, batch_size=batch_size, shuffle=shuffle)
    return generator

# 返回数据生成器
def load_tf_dataset (cfg):
    # 得到数据生成器
    generator = load_data_generator(cfg)
    # 批次大小
    batch_size = cfg.SOLVER.BATCH_SIZE
    # 定义数据tf生成器
    tf_dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32, tf.float32, tf.int32))
    # shuffle数据打乱
    tf_dataset = tf_dataset.shuffle(100*batch_size)
    # prefetch提供了一个软件pipelining操作机制，dataset可以预加载一些数据，加快模型训练速度
    tf_dataset = tf_dataset.prefetch(10*batch_size)
    # 一个批次的数据都填充为相同的shape
    # 图片数据，图片的元数据，图片中的标注框坐标，标注框标签
    # img[None, None, None], img_meta[None], bboxes[None, None], labels[None]
    tf_dataset = tf_dataset.padded_batch(
        batch_size, padded_shapes=([None, None, None], [None], [None, None], [None]))
    # 生成无数个批次
    tf_dataset = tf_dataset.repeat()
    return tf_dataset

# voc数据集类别
def voc_categories():
    classes = ('__background__', 
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')
    return classes

# coco数据集类别
def coco_categories():
    classes = ('__background__', 
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet',
                'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    return classes
        

