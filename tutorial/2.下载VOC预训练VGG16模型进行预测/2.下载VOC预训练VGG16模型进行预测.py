
# coding: utf-8

# In[6]:


from detecting.build.fasterrcnn import FasterRCNNModel
from detecting.config import cfg


# In[7]:


# cfg.merge_from_file('test.yml')使用'test.yml'修改默认配置
cfg.merge_from_file('test.yml')
# 然后把新的cfg传给模型
# 下载并载入预训练模型
model = FasterRCNNModel(cfg)


# In[9]:


# 预测结果并显示
model.predict_show('../../test_images/000000018380.jpg')

