import os
import numpy as np
from detecting.utils.logger import setup_logger
from detecting.config import cfg
from detecting.utils import visualize
from detecting.datasets.datasets import coco_categories,voc_categories
from detecting.models.detectors import faster_rcnn
from detecting.solver import *
from detecting.datasets.utils import load_img
import tensorflow as tf

class FasterRCNNModel():
    def __init__(self, config=None, 
                       backbone='resnet101', 
                       weights='COCO', 
                       input_shape=(1024,1024), 
                       **kwags):
        # 模型
        self.backbone = backbone
        # 权值
        self.weights = weights
        # 模型输入的图片大小
        self.input_shape = input_shape
        # 描述
        self.description = "Faster-RCNN"
        # 配置，如果传入了cfg参数，backbone和weights都以传入的cfg为准
        if config is not None:
            self.cfg = config
            self.backbone = self.cfg.MODEL.BACKBONE 
            self.weights = cfg.MODEL.WEIGHTS
            self.input_shape = cfg.MODEL.INPUT_SHAPE
        else:
            self.cfg = cfg
        # 固定配置内容
        self.cfg.freeze()
        # 数据集设置
        if self.cfg.DATASETS.NAMES == 'VOC':
            self.classes = voc_categories()
        elif self.cfg.DATASETS.NAMES == 'COCO':
            self.classes = coco_categories()
        else:
            self.classes = self.cfg.DATASETS.CLASSES

        # 不同模型创建方式
        if self.backbone=='resnet50' and self.weights=='COCO' and self.input_shape==(640,640):
            model_name = 'fasterrcnn_resnet50_640'
            base_url = 'http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200727093036-qyhky8vjois40s8o?attname=fasterrcnn_resnet50_640.zip&e=1595860441&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:y_jGmVl-uexuoKYbQJkcvOAJkK4='
            # build模型
            self.model = self._load_model(model_name, base_url, [640,640,3])
        elif self.backbone=='resnet50' and self.weights=='COCO' and self.input_shape==(1024,1024):
            model_name = 'fasterrcnn_resnet50_1024'
            base_url = 'http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200728094544-j9vusnec9uokk8sw?attname=fasterrcnn_resnet50_1024.zip&e=1595904466&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:8aMMMnTS3gsWr_d3KbRS07Hc2DM='
            # build模型
            self.model = self._load_model(model_name, base_url, [1024,1024,3])
        elif self.backbone=='resnet101' and self.weights=='COCO' and self.input_shape==(640,640):
            model_name = 'fasterrcnn_resnet101_640'
            base_url = 'http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200728095345-mild1p6duzkg80ww?attname=fasterrcnn_resnet101_640.zip&e=1595905768&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:JMGAxZR9RAiNz4CRc1bld2UgX3E='
            # build模型
            self.model = self._load_model(model_name, base_url, [640,640,3])    
        elif  self.backbone=='resnet101' and self.weights=='COCO' and self.input_shape==(1024,1024):
            model_name = 'fasterrcnn_resnet101_1024'
            base_url = 'http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200724041434-5aopoxzxgc4coc80?attname=fasterrcnn_resnet101_1024.zip&e=1595582245&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:8AKzHeuM0vBUfWplGOPzHTVe4BQ='
            # build模型
            self.model = self._load_model(model_name, base_url, [1024,1024,3])
        elif  self.backbone=='resnet152' and self.weights=='COCO' and self.input_shape==(640,640):
            model_name = 'fasterrcnn_resnet152_640'
            base_url = 'http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200728105534-ijwagx1ofxw8044w?attname=fasterrcnn_resnet152_640.zip&e=1595908752&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:oo0SpAvx3496ZqPMCDG6UkHlQ7k='
            # build模型
            self.model = self._load_model(model_name, base_url, [640,640,3])
        elif  self.backbone=='resnet152' and self.weights=='COCO' and self.input_shape==(1024,1024):
            model_name = 'fasterrcnn_resnet152_1024'
            base_url = 'http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200728110123-96bcwki8qfoc4gco?attname=fasterrcnn_resnet152_1024.zip&e=1595909196&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:rEiMY--bK2VkoPFAOn7Kgc5pLdE='
            # build模型
            self.model = self._load_model(model_name, base_url, [1024,1024,3])
        elif  self.backbone=='vgg16' and self.weights=='VOC' and self.input_shape==(1024,1024):
            model_name = 'fasterrcnn_vgg16_1024_VOC'
            base_url = 'http://ese5a4b0c7d11x.pri.qiqiuyun.net/attachment-3/20200730101828-39q2dxxb292ccwk4?attname=fasterrcnn_vgg16_1024_VOC.zip&e=1596190406&token=ExRD5wolmUnwwITVeSEXDQXizfxTRp7vnaMKJbO-:2H1LPFT0GEIwqU-sBqg4yGNWXm4='
            # build模型
            self.model = self._load_model(model_name, base_url, [1024,1024,3])

    
        # 创建一个没有训练过的新模型
        elif self.weights == 'None':
            # build模型
            self.model = self._build_model()
            print('build FasterRCNN')
        else:
            # build模型
            self.model = self._build_model()
            print('build FasterRCNN')
            # 载入训练好的模型的权值
            self.model.load_weights(self.weights, by_name=True, skip_mismatch=True)
            print('load weights from {}'.format(self.weights))
             
    # 创建模型
    def _build_model(self, image_shape=None):
        if image_shape is not None:
            shape = image_shape
        else:
            # 配置文件中的图片大小
            shape = [self.cfg.DATASETS.SCALE[0], self.cfg.DATASETS.SCALE[1], 3]
        # 定义模型
        model = faster_rcnn.FasterRCNN(self.cfg, len(self.classes))
        # 创建一个假的img用于build模型
        img = np.ones(shape)
        # build模型
        _ = model.predict(img)
        return model

    # 载入模型
    def _load_model(self, model_name, base_url, image_shape=None):
        model_dir = tf.keras.utils.get_file(
            fname = model_name + '.zip', 
            origin = base_url,
            extract = True)
        print('load ' + model_name + ' model')
        # 得到模型所在路径
        model_path = os.path.join(model_dir[:-4], model_name + '.h5')
        # 创建模型
        model = self._build_model(image_shape)
        # 载入训练好的模型的权值
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        return model

    def fit(self, generator):
        # 如果模型输出文件夹不存在则创建
        if cfg.OUTPUT_MODEL_DIR and not os.path.exists(cfg.OUTPUT_MODEL_DIR):
            os.makedirs(cfg.OUTPUT_MODEL_DIR)
        # 定义log文件位置
        log_dir = self.cfg.LOG_DIR
        # 如果log_dir文件夹不存在则创建
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # 定义logger
        self.logger = setup_logger(self.description, log_dir, 0)
        # 把配置内容写入log文件
        self.logger.info(self.cfg)
        # 设置优化器
        optimizer = set_optimizer(self.cfg)
        # 总的训练次数
        total_steps = self.cfg.SOLVER.TOTAL_STEPS
        # loss记录
        total_loss_history = []
        rpn_class_loss_h=[]
        rpn_bbox_loss_h=[]
        rcnn_class_loss_h=[]
        rcnn_bbox_loss_h=[]
        steps = 0
        for inputs in generator:
            steps += 1
            # 图片数据，图片的元数据，图片中的标注框坐标，标注框标签
            batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
            with tf.GradientTape() as tape:
                # 传入数据计算loss
                rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = \
                    self.model._compute_losses((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)
                # loss总和
                total_loss = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss
            # 计算梯度
            grads = tape.gradient(total_loss, self.model.trainable_variables)
            # 
            # grads = grads/self.cfg.SOLVER.BATCH_SIZE
            # 更新权值
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            # 保存loss
            total_loss_history.append(total_loss.numpy())
            rpn_class_loss_h.append(rpn_class_loss.numpy())
            rpn_bbox_loss_h.append(rpn_bbox_loss.numpy())
            rcnn_class_loss_h.append(rcnn_class_loss.numpy())
            rcnn_bbox_loss_h.append(rcnn_bbox_loss.numpy())
            # 每隔一段时间打印一次loss
            if steps % self.cfg.SOLVER.LOG_PERIOD == 0:
                # 打印一下loss变化
                self.logger.info('steps:{}, total_loss:{}'.format(steps,np.mean(total_loss_history)))
                self.logger.info('rpn_class_loss:{}'.format(np.mean(rpn_class_loss_h)))
                self.logger.info('rpn_bbox_loss:{}'.format(np.mean(rpn_bbox_loss_h)))
                self.logger.info('rcnn_class_loss:{}'.format(np.mean(rcnn_class_loss_h)))
                self.logger.info('rcnn_bbox_loss:{}'.format(np.mean(rcnn_bbox_loss_h)))
                total_loss_history = []
                rpn_class_loss_h=[]
                rpn_bbox_loss_h=[]
                rcnn_class_loss_h=[]
                rcnn_bbox_loss_h=[]
            # 每隔一段时间保存一次模型
            if steps % self.cfg.SOLVER.SAVE_MODEL == 0:
                # 保存模型
                self.model.save_weights(cfg.OUTPUT_MODEL_DIR+'faster_rcnn_{}.h5'.format(steps))
            # 达到预定训练次数，训练结束
            if steps >= total_steps:
                # 保存模型
                self.model.save_weights(cfg.OUTPUT_MODEL_DIR+'faster_rcnn_{}.h5'.format(steps))
                self.logger.debug('Finish training')
                return 1
       
    # 预测
    def predict(self, inputs, img_metas=None, box_mapping_back=True):
        # 如果传入一个路径则根据路径读取图片
        if isinstance(inputs, str):
            inputs = load_img(inputs)
        return self.model.predict(inputs, img_metas)

    # 预测并显示结果，一次只能传入一张图片
    def predict_show(self, inputs):
        # 如果传入一个路径则根据路径读取图片
        if isinstance(inputs, str):
            inputs = load_img(inputs)
        pre = self.model.predict(inputs)[0]
        # 显示结果
        visualize.display_instances(image=inputs, 
                                    boxes=pre['rois'], 
                                    class_ids=pre['class_ids'], 
                                    class_names=self.classes, 
                                    scores=pre['scores'])