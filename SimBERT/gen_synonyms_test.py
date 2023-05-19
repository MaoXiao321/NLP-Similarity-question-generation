#! -*- coding: utf-8 -*-
# SimBERT base 基本例子
# 环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.7.7

import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
from bert4keras.snippets import uniout
from keras.layers import *
import time

maxlen = 32

# bert配置
config_path = 'chinese_simbert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_simbert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)
encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate(
            [segment_ids, np.ones_like(output_ids)], 1)
        return seq2seq.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=5):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        
        output_ids = self.random_sample([token_ids, segment_ids], n, topk)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(start_id=None,
                                       end_id=tokenizer._token_end_id,
                                       maxlen=maxlen)


def gen_synonyms(text, n=100, k=20):
    """"含义： 产生text的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    """
    r = synonyms_generator.generate(text, n) # 生成n个相似句
    r = [i for i in set(r) if i != text] 
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x) # token id
        S.append(s) # segment id
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S]) # r中所有句子的句向量矩阵
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5 
    argsort = np.dot(Z[1:], -Z[0]).argsort() # 句子间的相似度矩阵
    return [r[i + 1] for i in argsort[:k]]


start = time.time()
print(gen_synonyms(u'无冷热选择权'))
end = time.time()
print(f'用时：{round(end-start)}s.')

"""
['无冷热选择权的选择', '无冷热选择权是什么', '什么是无冷热选择权', '无冷热选择权的概念', '如何选择无冷热选择权', '什么叫无冷热选择权', '无冷热选择权
的问题', '什么是无冷热选择权？', '无冷热选择权是什么？', '无冷热选择权的意思', '无冷热选择权，是否有用', '什么叫做无冷热选择权', '如何判断无冷热选择 
权', '冷热选择权', '无冷热选择权的概念是什么', '无冷热选择权是什么意思', '为什么说无冷热选择权', '无冷热选择权是什么意思?', '无冷热选择权是什么意思？
', '无冷热选择权是怎么回事']
用时：84s.
"""
