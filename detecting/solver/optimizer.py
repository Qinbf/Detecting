from detecting.solver.lr_schedules import set_lr_schedules
import tensorflow as tf
import numpy as np

# 设置优化器策略
def set_optimizer(cfg):
    lr_schedules = set_lr_schedules(cfg)
    if cfg.SOLVER.OPTIMIZER == 'rms_prop':
        optimizer = tf.keras.optimizers.RMSprop(
                                        lr_schedules,
                                        decay=cfg.SOLVER.OPTIMIZER_DECAY,
                                        momentum=cfg.SOLVER.OPTIMIZER_MOMENTUM,
                                        epsilon=cfg.SOLVER.OPTIMIZER_EPSILON)
    elif cfg.SOLVER.OPTIMIZER == 'momentum':
        optimizer = tf.keras.optimizers.SGD(
                                        lr_schedules, 
                                        momentum=cfg.SOLVER.OPTIMIZER_MOMENTUM, 
                                        nesterov=cfg.SOLVER.OPTIMIZER_NESTEROV)
    elif cfg.SOLVER.OPTIMIZER == 'adam':
        optimizer = tf.keras.optimizers.Adam(
                                        lr_schedules, 
                                        epsilon=cfg.SOLVER.OPTIMIZER_EPSILON)
    return optimizer

