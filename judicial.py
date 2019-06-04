# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:53:05 2019

@author: Administrator
"""
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import pickle
from gensim.models import KeyedVectors
from generateWordVec import *

dataPath = "E:\\Competition\\Dalian_bigdata\\contest_data\\judicial_data\\";


def SentencesVectore(sentence,size):
    """将已分词的句子更具词向量转化成embedding向量"""
    words = sentence.split(',');
    embedding = np.zeros(size);
    valid_words = 0;
    for word in words:
        if(word in model):
            embedding += model[word];
            valid_words += 1;
        else:
            continue;
    return embedding/valid_words if valid_words>0 else embedding;

def DataFrame2List(df,cols=None):
    if(cols is None):
        cols = df.columns.values.tolist()
        values = df.values.tolist();
    else:
        valid_index = [df.columns.get_loc(c) for c in cols]
        values = df.values[:,valid_index].tolist();
    
    return values;

def changeDataType(sentence_vec):
    if(isinstance(sentence_vec,str)):
        sentence_vec = sentence_vec.replace('\r','');
        sentence_vec = sentence_vec.replace('\n','');
        sentence_vec = sentence_vec[1:-1];
        
        sentence_vec = sentence_vec.split(' ');
        sentence_vec = [float(n) for n in sentence_vec if len(n)>0];
    
    return np.array(sentence_vec);

#准备罪名编号与内容对应字典
accus = pd.read_csv(dataPath + "accusation.csv");
index2accu = {};
accu2index = {}
for index,row in accus.iterrows():
    index2accu[row['index']] = row['accusation'];
    accu2index[row['accusation']] = row['index'];


def changeLabel(accuation):
    accuations = accuation.split(';');
    codes = [];
    for ac in accuations:
        codes.append(accu2index[ac]);
    return codes;


#词向量生成
if(os.path.exists(dataPath + "train_new.csv")):
    train = pd.read_csv(dataPath + "train_new.csv",usecols=['words','accuation']);
else:
    train = pd.read_csv(dataPath + "train.csv",usecols=['fact','accuation']);
    train['fact_chinese'] = train['fact'].map(removeUnchinese);
    train['words'] = train['fact_chinese'].map(cutWord);
#texts = creatCorproa(train['words'].values);

#加载词向量模型 Key_Value
model = KeyedVectors.load_word2vec_format(dataPath + "wordVector\\judicial.model_50_win7.bin",binary=True);
train = train.fillna(value={'words':' '});
#label = pd.read_csv(dataPath + "label.csv");
#train = train.merge(label,on='ids',how='left');
#train.to_csv(dataPath + "train_new.csv")
train['sentence_vec_50_win7'] = train['words'].apply(SentencesVectore,args=(50,));
train['codes'] = train['accusation'].map(changeLabel);

data = DataFrame2List(train,cols=['sentence_vec_50_win7','codes']);

#构建训练数据
N = len(data);
X = np.zeros((N,50));
Y = [];
#i = 0;
for i in range(N):
    row = data[i];
    X[i] = row[0];
#    c = row[1].replace('[','').replace(']','');
#    c = c.split(',');
#    c = [int(v) for v in c];
    Y.append(row[0]);
    #i = i + 1;
mb = MultiLabelBinarizer();
Y = mb.fit_transform(Y);
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=42);
#模型训练
print("begin to train");
lr = LogisticRegression(penalty='l1',tol=1e-5,random_state=47,solver='saga',max_iter=10000);
clf = OneVsRestClassifier(lr,n_jobs=-1);
clf.fit(X_train,y_train);
#模型测试
ty_pre = clf.predict(X_test);
print(metrics.classification_report(y_test,ty_pre));
print(np.mean(y_test==ty_pre));
pickle.dump(clf,open(dataPath+"model\\mutilclass_lr_50win70.pkl",'wb'));
print("model has been build");
#准备预测数据
if(os.path.exists(dataPath + "test_words.csv")):
    test = pd.read_csv(dataPath + "test_words.csv",usecols=['ids','words']);
else:
    test = pd.read_csv(dataPath + "test.csv",usecols=['ids','fact']);
    test['fact_chinese'] = test['fact'].map(removeUnchinese);
    test['words'] = test['fact_chinese'].map(cutWord);
    
test = test.fillna(value={'words':' '});
test['sentence_vec'] = test['words'].apply(SentencesVectore,args=(50,));
X = np.zeros((len(test),50));
for i in range(len(test)):
    X[i] = test['sentence_vec'].values[i];

#clf = pickle.load(open(dataPath + "mdoel\\mutilclass_lr.pkl",'rb'));
#预测
print("begin to predict")
y_pred = clf.predict(X);
y_prob = clf.predict_proba(X);


#结果转化
result = [];
for i,y in enumerate(y_pred):
    r = [];
    indexs = np.where(y==1)[0];
    if(len(indexs)==0):
        index = np.argmax(y_prob[i]);
        r.append(index2accu[index]);
    else:
        for index in indexs:
            r.append(index2accu[index]);
    result.append(';'.join(r));
    
ids = test['ids'].values;
#预测结果保存
f = open(dataPath + "submit\\submit4.csv",mode='w',encoding='utf-8');
f.write("submit4\n");
f.write("ids,accusation\n");
for ids,accu in zip(ids,result):
    f.write(ids+","+accu+'\n');
f.close();

print("Done");
