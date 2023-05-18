# 利用浏览器生成相似问并算句子相似度得分

## 相似问生成
目前电商平台的可参考淘宝的API接口，地址为:https://suggest.taobao.com/sug?area=etao&code=utf-8&callback=KISSY.Suggest.callback&q=需要扩写的query<br>
百度的：http://suggestion.baidu.com/su?wd=需要扩写的query&p=3&cb=window.bdsug.sug或是https://www.baidu.com/sugrec?pre=1&p=3&ie=utf-8&json=1&prod=pc&from=pc_web&wd=需要扩写的query<br>
搜狗的：http://w.sugg.sogou.com/sugg/ajaj_json.jsp?key=需要扩写的query&type=web<br>
必应的：http://sg1.api.bing.com/qsonhs.aspx?type=cb&cb=callback&q=需要扩写的query<br>

参考：https://zhuanlan.zhihu.com/p/525977098

## 算句子相似度
利用hugging face模型库中的sbert-base-chinese-nli算句子相似度。句子的表征向量是平均所有位置的输出值得到。