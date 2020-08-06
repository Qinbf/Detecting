
# coding: utf-8

# In[ ]:


from detecting.build.fasterrcnn import FasterRCNNModel
from detecting.datasets.datasets import load_tf_dataset
from detecting.config import cfg


# In[ ]:


# 与配置文件中的配置合并
cfg.merge_from_file('train.yml')
# 载入数据集tf_dataset
tf_dataset = load_tf_dataset(cfg)
# 载入模型 
model = FasterRCNNModel(cfg)
# 训练模型
model.fit(tf_dataset)

