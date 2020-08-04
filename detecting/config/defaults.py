# pip install yacs
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# 基本配置
# -----------------------------------------------------------------------------
_C = CN()
# log输出文件夹
_C.LOG_DIR = "log"
# 输出训练得到的模型位置
_C.OUTPUT_MODEL_DIR = 'training/'
# -----------------------------------------------------------------------------
# 模型配置
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# 基本结构
# vgg16
# resnet50/resnet101/resnet152
_C.MODEL.BACKBONE = 'resnet101'
# 模型输入
_C.MODEL.INPUT_SHAPE = (1024,1024)
# 预训练的目标检测模型位置
# 'None'表示没有预训练模型
# 也可以是预训练模型的具体地址
# 也可以设置为'COCO'或'VOC'表示使用'COCO'或'VOC'数据集训练得到的模型
_C.MODEL.WEIGHTS = 'None'
# 注意这里创建detection模型时设置的分类类别数
# 表示detection模型最后的分类预测有NUM_CLASSES个结果
# 并不一定是数据集中真实的分类类别数
_C.MODEL.NUM_CLASSES = 21
# anchor尺寸
_C.MODEL.ANCHOR_SCALES= (64, 128, 256, 512)
# anchor比例
_C.MODEL.ANCHOR_RATIOS = (0.5, 1, 2)
# 特征图对应原始图片的缩放比例
_C.MODEL.ANCHOR_FEATURE_STRIDES = (16, 16, 16, 16)
#_C.MODEL.ANCHOR_FEATURE_STRIDES = (4, 8, 16, 32, 64)
# 调整RPN层候选框时使用的均值
_C.MODEL.RPN_TARGET_MEANS = (0., 0., 0., 0.)
# 调整RPN层候选框时使用的标准差
_C.MODEL.RPN_TARGET_STDS = (0.1, 0.1, 0.2, 0.2)
# RPN层训练批次大小
_C.MODEL.RPN_BATCH_SIZE = 256
# RPN层正样本比例
_C.MODEL.RPN_POS_FRAC = 0.5
# RPN正样本IOU阈值
_C.MODEL.RPN_POS_IOU_THR = 0.7
# RPN负样本IOU阈值
_C.MODEL.RPN_NEG_IOU_THR = 0.3
# 对前N个概率最大的anchors进行NMS
_C.MODEL.RPN_PROPOSAL_NMS_TOP_N = 12000
# ROI数量，候选区域数量
_C.MODEL.RPN_PROPOSAL_COUNT = 2000
# 非极大值抑制阈值
_C.MODEL.RPN_NMS_THRESHOLD = 0.7
# ROIAlign层使用的固定特征图大小
_C.MODEL.POOL_SIZE = (14, 14)
# 调整RCNN预测框时使用的均值
_C.MODEL.RCNN_TARGET_MEANS = (0., 0., 0., 0.)
# 调整RCNN预测框时使用的标准差
_C.MODEL.RCNN_TARGET_STDS = (0.1, 0.1, 0.2, 0.2)
# RCNN批次大小
_C.MODEL.RCNN_BATCH_SIZE = 256
# RCNN正样本比例
_C.MODEL.RCNN_POS_FRAC = 0.25
# RCNN正样本IOU阈值
_C.MODEL.RCNN_POS_IOU_THR = 0.5
# RCNN负样本IOU上阈值
_C.MODEL.RCNN_NEG_IOU_THR_HIGH = 0.5
# RCNN负样本IOU下阈值
_C.MODEL.RCNN_NEG_IOU_THR_LOW = 0.1
# RCNN置信度阈值
_C.MODEL.RCNN_MIN_CONFIDENCE = 0.1
# RCNN预测框非极大值抑制阈值
_C.MODEL.RCNN_NMS_THRESHOLD = 0.5
# 最多预测框数量
_C.MODEL.RCNN_MAX_INSTANCES = 100
# 是否所有种类的共享回归预测结果
_C.MODEL.SHARE_BOX_ACROSS_CLASSES = True
# -----------------------------------------------------------------------------
# 数据集配置
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# 数据集'VOC'或者'COCO'
_C.DATASETS.NAMES = ('COCO')
# 数据集根目录
_C.DATASETS.ROOT_DIR = ('./datasets/COCO2017')
# 'train','test' or 'val'
_C.DATASETS.SUBSET = ('val')
# 自定义数据类别
# 搭建模型的时候会根据类别数量来定义最后的分类数量
_C.DATASETS.CLASSES = ('__background__','class1','class2')
# 图片保存路径
_C.DATASETS.IMAGE_DIR = ''
# 标注路径
_C.DATASETS.LABEL_DIR = ''
# 图片翻转概率
_C.DATASETS.FLIP_RATIO = 0.5
# 图片填充方式'fixed', 'non-fixed'
_C.DATASETS.PAD_MODE = 'fixed'
# 图片缩放是否保持宽高比
# 比如设置为True，SCALE=(1024,800)
# 图片大小，原始图片会等比例缩放
# 长边缩放到1024，并且短边不超过800
# 或者短边缩放到800，长边不超过1024
_C.DATASETS.KEEP_ASPECT = False
# 图像均值
_C.DATASETS.IMG_MEAN = (123.68,116.779,103.939)
# 图像标准差
_C.DATASETS.IMG_STD = (1., 1., 1.)
# 设置图片高度和宽度
_C.DATASETS.SCALE = (1024, 1024)
# -----------------------------------------------------------------------------
# 模型训练配置
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
# 总的训练次数
_C.SOLVER.TOTAL_STEPS = 100000
# 批次大小
_C.SOLVER.BATCH_SIZE = 1
# 基本学习率
_C.SOLVER.BASE_LR = 3e-3
# 'rms_prop'
# 'momentum'
# 'adam'
_C.SOLVER.OPTIMIZER = 'momentum'
# momentum
_C.SOLVER.OPTIMIZER_NESTEROV = False
# momentum/rms_prop
_C.SOLVER.OPTIMIZER_MOMENTUM = 0.9
# rms_prop
_C.SOLVER.OPTIMIZER_DECAY = 0.9
# rms_prop/adam,1/1e-8
_C.SOLVER.OPTIMIZER_EPSILON = 1
# 'exponential_decay'
# 'constant_decay'
# 'cosine_decay'
# 'cosine_decay_with_warmup'
_C.SOLVER.LR_SCHEDULES = 'cosine_decay_with_warmup'
# PiecewiseConstantDecay学习率策略参数
_C.SOLVER.BOUNDARIES = [30000,60000]
# PiecewiseConstantDecay学习率策略参数
_C.SOLVER.LR_VALUES = [1e-4, 1e-5, 1e-6]
# 每多少steps进行学习率调整
_C.SOLVER.DECAY_STEPS = 30000
# 学习率衰减率
_C.SOLVER.DECAY_RATE = 0.1
# 在ExponentialDecay学习率策略中有效，是否阶梯下降
_C.SOLVER.STAIRCASE = True
# warmup学习率
_C.SOLVER.WARMUP_LEARNING_RATE = 1e-3
# warmup步数
_C.SOLVER.WARMUP_STEPS = 2000
# 基本学习率保持的步数
_C.SOLVER.HOLD_BASE_RATE_STEPS = 0
# 多少周期保存一次模型
_C.SOLVER.SAVE_MODEL = 1000
# 训练多少次进行显示一次训练log
_C.SOLVER.LOG_PERIOD = 20
# 训练多少次进行一次模型测试
_C.SOLVER.EVAL_PERIOD = 50
# 是否训练BN
_C.SOLVER.BN_TRAIN = True
# 权值衰减
_C.SOLVER.WEIGHT_DECAY = 0
