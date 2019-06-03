# PredictCharges
this is a competition of Dalian University    
## 项目简介    
训练集大约37万条数据，测试集大约17万条数据    
训练集包含 ids,fact,criminals,articles字段，测试集包含ids,fact字段，具体字段详见  [说明](https://github.com/firewolffff/PredictCharges/blob/master/%E8%AF%B4%E6%98%8E.docx)  
罪名域是犯罪频率最高的30中罪名，从训练集中将罪名抽取出来，按照频数从高到低排列，并对其进行重新编号，结果保存到accasation.csv文件中。    
## 建模过程
1. 去除非中文字符    
2. 分词，保留n 名词, v 动词 , a 形容词 , m 数词 ,q 量词 , d 副词     
3. 生成预料文本，生成词向量模型    
4. 依据犯罪事实的分词结果得到犯罪事实的文本向量，采用embedding方式生成犯罪事实文本向量   
5. 使用logisticRegression模型作为预测模型，使用OneVsRestClassifier训练出30个二分类模型     
具体建模流程详见  [generateWordVec](https://github.com/firewolffff/PredictCharges/blob/master/generateWordVec.py) ,[judicial](https://github.com/firewolffff/PredictCharges/blob/master/judicial.py)    

## 模型效果    
![model effects](https://github.com/firewolffff/PredictCharges/blob/master/mutilclass_lr_80.pkl.png)    
## 模型测试集成绩    
![model score](https://github.com/firewolffff/PredictCharges/blob/master/score.jpg)    
