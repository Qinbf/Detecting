
# coding: utf-8

# In[1]:


from detecting.build.fasterrcnn import FasterRCNNModel


# In[2]:


# 下载并载入预训练模型
model = FasterRCNNModel(backbone='resnet101', weights='COCO', input_shape=(1024, 1024))


# In[3]:


# 预测结果并显示
model.predict_show('../../test_images/000000018380.jpg')


# In[4]:


# 预测结果
model.predict('../../test_images/000000018380.jpg')

