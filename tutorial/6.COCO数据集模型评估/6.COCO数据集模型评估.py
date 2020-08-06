
# coding: utf-8

# In[1]:


from detecting.config import cfg
from detecting.build.fasterrcnn import FasterRCNNModel
from detecting.datasets.datasets import load_data_generator
from detecting.utils.eval_coco import eval_coco


# In[ ]:


# 与配置文件中的配置合并
cfg.merge_from_file('eval.yml')
# 载入模型
model = FasterRCNNModel(cfg)
# 数据生成器
generator = load_data_generator(cfg)
# 评估结果
eval_coco(model, generator)

