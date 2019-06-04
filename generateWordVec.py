# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:17:26 2019

@author: Administrator
"""
import jieba
from jieba import analyse as ana
from jieba import posseg as pseg
from collections import defaultdict
import pandas as pd
import re
import os

from gensim import corpora
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

__all__ = ['removeUnchinese','cutWord'];

def removeUnchinese(fact):
    """
    剔除非汉字和长度小于等于2的汉字或者词
    """
    re_word = re.compile(u"[\u4e00-\u9fa5]+");
    result = re_word.findall(fact);
    result = [r for r in result if len(r)>2];
    return ','.join(result);

def cutWord(sentence):
    """抽取长度大于1 的名词，动词，形容词，量词，数词，副词"""
    res = [];
    words = pseg.cut(sentence);
    for w , f in words:
        if(len(w)>1 and f in ['n','v','a','m','q','d']):
            res.append(w);
    if(len(res)==0):
        return "nan";
    return ','.join(res);

"""
sentences list 已分词后的句子，单词间用 , 隔开
return list  单词词频大于1的单词的句子集合语料
"""
def creatCorproa(sentences):
    texts = [[word for word in sentence.split(',') ] for sentence in sentences];
    frequency = defaultdict(int);
    for text in texts:
        for token in text:
            frequency[token] += 1;

    texts = [[token for token in text if frequency[token] > 1] for text in texts];
    #dictionary = corpora.Dictionary(texts);
    #dictionary.save(dataPath + 'judicial_corproa.dict');
    return texts;

#词向量生成
if __name__ == "__main__":
    dataPath = "E:\\Competition\\Dalian_bigdata\\contest_data\\judicial_data\\";
    if(os.path.exists(dataPath + "train_new.csv")):
        train = pd.read_csv(dataPath + "train_new.csv",usecols=['words']);
        train = train.dropna();
    else:
        train = pd.read_csv(dataPath + "train.csv",usecols=['ids','fact','accusation']);
        train = train.drop_duplicates(subset=['ids']);
        train['fact_chinese'] = train['fact'].map(removeUnchinese);
        train['words'] = train['fact_chinese'].map(cutWord);
        train = train.dropna(subset=['words']);
    texts = creatCorproa(train['words'].values);
    #列表应是一个二维数组，每一行表示一个句子词组
    #它认为列表中的每个单词b都是一个句子,因此它为每个单词中的每个单词执行Word2Vec,而不是b中的每个单词。 
    model = Word2Vec(texts,min_count=1,size=50,window=5);
    model.wv.save_word2vec_format(dataPath + 'wordVector\\judicial.model_size50.bin', binary=True);
    
    if(not os.path.exists(dataPath + "train_new.csv")):
        train.to_csv(dataPath + "train_new.csv",index=False,encoding='utf-8');
