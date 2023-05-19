# RoFormer-Sim
RoFormer-Sim，又称SimBERTv2，是SimBERT模型的升级版。原项目地址：https://github.com/ZhuiyiTechnology/roformer-sim

## 环境
tensorflow 1.14 + keras 2.3.1 + bert4keras 0.10.6

## 权重下载
- [chinese_roformer-sim-char_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char_L-12_H-768_A-12.zip)
- [chinese_roformer-sim-char_L-6_H-384_A-6.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char_L-6_H-384_A-6.zip)
- [chinese_roformer-sim-char-ft_L-12_H-768_A-12.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char-ft_L-12_H-768_A-12.zip)
- [chinese_roformer-sim-char-ft_L-6_H-384_A-6.zip](https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roformer-sim-char-ft_L-6_H-384_A-6.zip)

## 生成相似问
```
pyhton gen_synonyms_test.py
```

## 算两个句子之间的相似度
```
pyhton retrieval.py
```
