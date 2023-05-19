#! -*- coding: utf-8 -*-
# RoFormer-Sim base 基本例子
# 测试环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.10.6

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
from bert4keras.snippets import uniout
import time

maxlen = 64

# 模型配置
config_path = 'chinese_roformer-sim-char_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_roformer-sim-char_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_roformer-sim-char_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
roformer = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roformer',
    application='unilm',
    with_pool='linear'
)

encoder = keras.models.Model(roformer.inputs, roformer.outputs[0])
seq2seq = keras.models.Model(roformer.inputs, roformer.outputs[1])


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(seq2seq).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95, mask_idxs=[]):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        for i in mask_idxs:
            token_ids[i] = tokenizer._token_mask_id
        output_ids = self.random_sample([token_ids, segment_ids], n,
                                        topp=topp)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)


def gen_synonyms(text, n=100, k=20, mask_idxs=[]):
    ''''含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    '''
    r = synonyms_generator.generate(text, n, mask_idxs=mask_idxs)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]


start = time.time()
result = gen_synonyms(u'无冷热选择权')
end = time.time()
print(result)
print(f'用时：{round(end-start)}s.')

"""
['无冷热选择权：', '无冷热选择权:', '无冷热选择权，冷热选择', '无无冷热选择权', '无冷热选择权冷热选择', '无冷热选择权。', '选择权：无冷热选择权', '无
冷热选择权|', '无冷热和冷热选择权', '无冷热选择权！', '是无冷热选择权。', '无机冷热选择权', '有冷热选择权', '无冷热选择权系统', '无冷热选择权相关信息
', '冷热选择权', '您好无冷热选择权', '没有冷热选择权', '无冷热模式选择权', '无冷热结构选择权']
用时：79s.
"""
