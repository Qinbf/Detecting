
# coding: utf-8

# In[1]:


from detecting.build.fasterrcnn import FasterRCNNModel
from detecting.datasets.datasets import load_data_generator
from detecting.datasets.utils import get_original_image
from detecting.config import cfg


# In[2]:


# cfg.merge_from_file('test.yml')使用'test.yml'修改默认配置
cfg.merge_from_file('test.yml')
# 然后把新的cfg传给模型
model = FasterRCNNModel(cfg)


# In[3]:


# 得到数据生成器
generator = load_data_generator(cfg)


# In[5]:


# 循环2个批次数据
for i in range(10, 12):
    # 生成一个批次数据，默认的批次大小为1
    batch_imgs, batch_metas, batch_bboxes, batch_labels = generator[i]
    # 得到原始图片（生成器产生的数据会进行数据标准化处理）
    ori_imgs = get_original_image(batch_imgs, batch_metas)
    # 预测并显示结果
    model.predict_show(ori_imgs)

