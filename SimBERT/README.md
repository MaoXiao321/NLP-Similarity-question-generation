# SimBERT
基于UniLM思想、融检索与生成于一体的BERT模型。原项目地址：https://github.com/ZhuiyiTechnology/simbert

## 训练环境
tensorflow 1.14 + keras 2.3.1 + bert4keras 0.7.7

以windows为例：从https://pypi.org/project/tensorflow/1.14.0/#files下载tensorflow-1.14.0-cp37-cp37m-win_amd64.whl,pip默认会安装tf2.0以上版本
```
conda create -n tf1.14 python=3.7
conda activate tf1.14
pip install D:\softwares\tensorflow-1.14.0-cp37-cp37m-win_amd64.whl
pip install keras==2.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install bert4keras==0.9.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda list # 检查下
```

## 权重下载
https://github.com/ZhuiyiTechnology/pretrained-models
此处下载的是基于2200万相似句组训练得到的chinese_simbert_L-12_H-768_A-12。文件夹下包含：
-bert_config.json
-bert_model.ckpt.data-00000-of-00001
-bert_model.ckpt.index
-checkpoint
-vocab.txt

## SimBERT训练代码
```
pyhton simbert.py
```

## 生成相似问
```
pyhton gen_synonyms_test.py
```
bug修改：
（1）from bert4keras.backend import keras, K报错：Descriptors cannot not be created directly.
将protobuf版本降级：pip install protobuf==3.20.*
（2）type object 'AutoRegressiveDecoder' has no attribute 'set_rtype'
只是换了函数名，直接把 set_rtype 改成 wraps 就好了
（3）encode() got an unexpected keyword argument 'max_length'
max_length换成maxlen


## 在指定内容中找到最相近的topK句子
```
pyhton retrieval_test.py.py
```
