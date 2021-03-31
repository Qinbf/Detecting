import logging
import os
import sys

# 定义log生成
def setup_logger(name, save_dir, distributed_rank):
    level = logging.DEBUG
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if distributed_rank > 0:
        return logger
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    # 输出信息
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # 保存信息
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
