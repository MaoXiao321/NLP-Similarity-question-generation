"""
利用搜索平台扩展相似问，利用sbert-base-chinese-nli算句子的相似度
"""

import requests
import json
import pandas as pd

def get_similarity(sentences):
    from transformers import BertTokenizer, BertModel
    from sklearn.metrics.pairwise import cosine_similarity
    import torch

    model = BertModel.from_pretrained("uer/sbert-base-chinese-nli")
    tokenizer = BertTokenizer.from_pretrained("uer/sbert-base-chinese-nli")
    # sentences = [
    #     "那个人很开心",
    #     "那个人非常开心",
    #     "那只猫很开心",
    #     "那个人在吃东西"
    # ]

    # 初始化字典来存储
    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sentences:
        # 编码每个句子并添加到字典
        new_tokens = tokenizer.encode_plus(sentence, max_length=15, truncation=True, padding='max_length', return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # 将张量列表重新格式化为一个张量
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    outputs = model(**tokens)
    # print(outputs.keys()) #odict_keys(['last_hidden_state', 'pooler_output'])

    # 得到句子的向量表示，此处是平均所有位置的输出值mean_pooled
    embeddings = outputs.last_hidden_state
    # print(embeddings.shape) #torch.Size([4, 15, 768])

    attention_mask = tokens['attention_mask']
    # print(attention_mask.shape) #torch.Size([4, 15])
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    # print(mask.shape) #torch.Size([4, 15, 768])

    masked_embeddings = embeddings * mask
    # print(masked_embeddings.shape) #torch.Size([4, 15, 768])

    summed = torch.sum(masked_embeddings, 1)
    # print(summed.shape) #torch.Size([4, 768])

    # clamp将元素值压缩到指定区间
    summed_mask = torch.clamp(mask.sum(1), min=1e-9) 
    # print(summed_mask.shape) #torch.Size([4, 768])

    mean_pooled = summed / summed_mask
    # print(mean_pooled.shape) #torch.Size([4, 768])
    # detach()从原计算图中分离下来，生成新tensor
    mean_pooled = mean_pooled.detach().numpy() 

    # 算cosine_similarity算句子相似度
    result = cosine_similarity([mean_pooled[-1]], mean_pooled[:-1])
    # print(result) #[[0.9864919  0.39011386 0.29779416]]
    return result[0]

def get_simiquery_baidu(query):
    url = f"https://www.baidu.com/sugrec?pre=1&p=3&ie=utf-8&json=1&prod=pc&from=pc_web&wd={query}"
    result = requests.get(url).text
    try:
        res = json.loads(result)["g"]
        similarity_querys = [res[i]["q"] for i in range(len(res))]
    except:
        similarity_querys = []
    return similarity_querys

def get_simiquery_biying(query):
    url = f"http://sg1.api.bing.com/qsonhs.aspx?type=cb&cb=callback&q={query}"
    result = requests.get(url).text
    result = result.replace("if(typeof callback == \'function\') callback(","")
    result = result.replace("/* pageview_candidate */);","")
    try:
        res = json.loads(result)['AS']['Results'][0]['Suggests']
        similarity_querys = [res[i]['Txt'] for i in range(len(res))]
    except:
        similarity_querys = []
    return similarity_querys

if __name__ == "__main__":
    queries = ["故障代码速查","无冷热选择权","除湿-不启动","水龙头标志"]
    # queries = ["无冷热选择权"]
    output = pd.DataFrame()
    for query in queries:
        simiquery_baidu = get_simiquery_baidu(query)
        simiquery_biying = get_simiquery_biying(query)
        similarity_querys = list(set(simiquery_baidu).union(set(simiquery_biying)))
        if len(similarity_querys) > 0:
            sentences = similarity_querys.copy()
            sentences.append(query)
            similarity = get_similarity(sentences).tolist()
            
            df = pd.DataFrame()
            df["similarity_query"] = similarity_querys
            df["query"] = query
            df["similarity"] = similarity
            output = pd.concat([output,df],axis=0)  
    if len(output) > 0:      
        output = output[["query","similarity_query","similarity"]]
        output.to_excel("相似问生成.xlsx",index=False)