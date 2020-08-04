from detecting.models.backbones import VGG16,resnet_v1_50,resnet_v1_101,resnet_v1_152
from detecting.utils import model_util
from tensorflow.keras.models import Model

# 选择不同的backbone
def get_backbone(cfg):
    if cfg.MODEL.BACKBONE=='vgg16':
        return backbone_vgg16(cfg)

    elif cfg.MODEL.BACKBONE=='resnet50':
        return backbone_resnet50(cfg)

    elif cfg.MODEL.BACKBONE=='resnet101':
        return backbone_resnet101(cfg)
    
    elif cfg.MODEL.BACKBONE=='resnet152':
        return backbone_resnet152(cfg)

def backbone_vgg16(cfg):
    vgg16 = VGG16(include_top=True, weights='imagenet')
    # 不要最后的一个池化层
    backbone = Model(inputs=vgg16.input,outputs=vgg16.get_layer('block5_conv3').output)
    # conv3_1之前的层不训练
    for layer in backbone.layers[:7]:
        layer.trainable = False
    # 获取vgg16最后分类的那一部分
    head_to_tail = model_util.extract_submodel(
                                          model=vgg16,
                                          inputs=vgg16.get_layer('block5_pool').output,
                                          outputs=vgg16.get_layer('fc2').output) 
    return backbone, head_to_tail

def backbone_resnet50(cfg):
    resnet50 = resnet_v1_50(batchnorm_training=cfg.SOLVER.BN_TRAIN,
                             weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                             classes=None,
                             weights=None,
                             include_top=False)
    backbone = Model(inputs=resnet50.input,outputs=resnet50.get_layer('conv4_block6_out').output)
    head_to_tail = model_util.extract_submodel(
                                    model=resnet50,
                                    inputs=resnet50.get_layer('conv4_block6_out').output,
                                    outputs=resnet50.get_layer('conv5_block3_out').output) 
    return backbone, head_to_tail         

def backbone_resnet101(cfg):
    resnet101 = resnet_v1_101(batchnorm_training=cfg.SOLVER.BN_TRAIN,
                            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                            classes=None,
                            weights=None,
                            include_top=False)
    backbone = Model(inputs=resnet101.input,outputs=resnet101.get_layer('conv4_block23_out').output)
    head_to_tail = model_util.extract_submodel(
                                    model=resnet101,
                                    inputs=resnet101.get_layer('conv4_block23_out').output,
                                    outputs=resnet101.get_layer('conv5_block3_out').output) 
    return backbone, head_to_tail 

def backbone_resnet152(cfg):
    resnet152 = resnet_v1_152(batchnorm_training=cfg.SOLVER.BN_TRAIN,
                              weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                              classes=None,
                              weights=None,
                              include_top=False)
    backbone = Model(inputs=resnet152.input,outputs=resnet152.get_layer('conv4_block36_out').output)
    head_to_tail = model_util.extract_submodel(
                                    model=resnet152,
                                    inputs=resnet152.get_layer('conv4_block36_out').output,
                                    outputs=resnet152.get_layer('conv5_block3_out').output) 
    return backbone, head_to_tail 

