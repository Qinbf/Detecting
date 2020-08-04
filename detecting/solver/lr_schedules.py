import tensorflow as tf
import numpy as np

# 设置学习率策略
def set_lr_schedules(cfg):
    # 指数学习率策略
    if cfg.SOLVER.LR_SCHEDULES == 'exponential_decay':
        lr_schedules = exponential_decay(cfg)
    # 常数学习率策略
    elif cfg.SOLVER.LR_SCHEDULES == 'constant_decay':
        lr_schedules = constant_decay(cfg)
    # cosine学习率策略
    elif cfg.SOLVER.LR_SCHEDULES == 'cosine_decay':
        lr_schedules = cosine_decay(cfg)
    # 带warmup的cosine学习率测试
    elif cfg.SOLVER.LR_SCHEDULES == 'cosine_decay_with_warmup':
        lr_schedules = cosine_decay_with_warmup(cfg)
    return lr_schedules

# 指数学习率策略
def exponential_decay(cfg):
    lr_schedules = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=cfg.SOLVER.BASE_LR,
                                                                decay_steps=cfg.SOLVER.DECAY_STEPS,
                                                                decay_rate=cfg.SOLVER.DECAY_RATE,
                                                                staircase=True)
    return lr_schedules

# 常数学习率策略
def constant_decay(cfg):
    boundaries = cfg.SOLVER.BOUNDARIES
    values = cfg.SOLVER.LR_VALUES
    lr_schedules = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=values)
    return lr_schedules

# cosine学习率策略
class cosine_decay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, cfg):
        super().__init__()
        self.initial_learning_rate = tf.cast(cfg.SOLVER.BASE_LR, dtype=tf.float32)
        self.decay_steps = tf.cast(cfg.SOLVER.DECAY_STEPS, dtype=tf.float32)
    def __call__(self, step):
        return self.initial_learning_rate * (1 + tf.math.cos(step * np.pi / self.decay_steps)) / 2

# 带warmup的cosine学习率测试
class cosine_decay_with_warmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, cfg):
        # 基础学习率
        self.learning_rate_base = cfg.SOLVER.BASE_LR
        # 总的训练次数
        self.total_steps = cfg.SOLVER.TOTAL_STEPS
        # warmup学习率
        self.warmup_learning_rate = cfg.SOLVER.WARMUP_LEARNING_RATE
        # warmup步数
        self.warmup_steps = cfg.SOLVER.WARMUP_STEPS
        # 保持base_lr的步数
        self.hold_base_rate_steps = cfg.SOLVER.HOLD_BASE_RATE_STEPS
    def __call__(self, step):
        # 总的steps不能小于warmup_steps
        if self.total_steps < self.warmup_steps:
                raise ValueError('total_steps must be larger or equal to '
                                'warmup_steps.')
        # 先增大后减小的学习率
        learning_rate = 0.5 * self.learning_rate_base * (1 + tf.cos(
            np.pi *
            (tf.cast(step, tf.float32) - self.warmup_steps - self.hold_base_rate_steps
            ) / float(self.total_steps - self.warmup_steps - self.hold_base_rate_steps)))
        # 如果base_rate要保持一段时间
        if self.hold_base_rate_steps > 0:
            learning_rate = tf.where(
                step > self.warmup_steps + self.hold_base_rate_steps,
                learning_rate, self.learning_rate_base)
        # 进行warmup
        if self.warmup_steps > 0:
            # 基本学习率不能小于warmup的学习率
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to '
                                    'warmup_learning_rate.')
            slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
            # 计算当前warmup学习率
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(step < self.warmup_steps, warmup_rate, learning_rate)
        
        return tf.where(step > self.total_steps, 0.0, learning_rate)