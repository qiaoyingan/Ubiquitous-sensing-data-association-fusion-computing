# -*- coding: utf-8 -*-
"""
CCF2021 run 算法基准程序
@author: zhouwei
"""

import pandas as pd

import random
import numpy as np

from xgboost import XGBClassifier

#根据一定的特征时间初步关联图码数据-全关联
def get_MatchRecord(imsi,face, deltaSeconds):
    match_record2=[]
    for i in face.index:
        deviceID=face['DeviceID'][i]
        threshold2=deltaSeconds[deviceID]
        f=face['FaceLabel'][i]
        ts=face['TimeStamp'][i]
        match_imsi2=imsi[(imsi['DeviceID']==deviceID) & (abs(imsi['TimeStamp']-ts)<threshold2)]['Code']
        if len(match_imsi2)>0:
            for code in match_imsi2:
                match_record2.append([f,code,ts,deviceID])    
    df_Match2=pd.DataFrame(match_record2,columns=['FaceLabel','Code','TimeStamp','DeviceID'])
    return df_Match2

#基于图码预关联的数据，进行特征提取的准备工作    
def genFeature(df_Match, df_Face, df_Imsi, deltaSeconds):
    listDeviceID = list(deltaSeconds.keys())
    listDeviceID.sort()
        
    #图的总次数
    df_T_temp3 = df_Face.groupby('FaceLabel').count().reset_index()[['FaceLabel','DeviceID']]
    df_T_temp3.columns = ['FaceLabel','countT']
    
    df_T_count = df_T_temp3['countT']
    df_T_temp3['countT'] = df_T_count
    
    df_T = df_T_temp3
        
    #码的总次数
    df_M_temp3 = df_Imsi.groupby('Code').count().reset_index()[['Code','DeviceID']]
    df_M_temp3.columns = ['Code','countM']
    
    df_M_count = df_M_temp3['countM']
    df_M_temp3['countM'] = df_M_count
    
    df_M = df_M_temp3
    
    #图码关联的总次数
    df_TM_temp3 = df_Match.groupby(['FaceLabel','Code']).count().reset_index()[['FaceLabel','Code', 'DeviceID']]
    df_TM_temp3.columns = ['FaceLabel','Code','countTM']
    
    df_TM_count = df_TM_temp3['countTM']
    df_TM_temp3['countTM'] = df_TM_count
    
    df_TM = df_TM_temp3
        
    #汇总所有特征
    res = df_TM.merge(df_T,on='FaceLabel',how='left').merge(df_M, on='Code', how='left')
    return res

#标记图码关联，生成训练数据的label
def label(row):
    p = row['FaceLabel']
    c = row['Code']
    #if(c.endswith(p)):
    if dict_label.get(c) == p:
        return 1
    else:
        return 0

#根据FaceLabel，标记训练数据
def testflag(row):
    p = row['FaceLabel']
    h = hash(p)
    if h % 100 < 30: # 30%为测试集
        return 1
    else:
        return 0


#生成训练的基础数据    
def genData(df,multiple):
    df['label'] = df.apply(label,axis=1)    
    postive = df[df['label']==1]
    
    negative = df[df['label']==0].sample(multiple*len(postive), replace=False, weights="countTM")
    comb = pd.concat([postive,negative])
    #data = comb[comb.columns[2:]] #前两列 FaceLabel Code
    data = comb
    #X = np.array(data[data.columns[:-1]])
    #y = np.array(data['label'])
    X = data[data.columns[:-1]]
    y = data['label']
    return (X,y)
    
rdseed = 10
random.seed(rdseed)
np.random.seed(rdseed)

num_people =  2000

maxDeltaT = 120 #120秒为最长采集时差

train_folder, test_folder, result_folder = 'train_dataset/', 'test_dataset/', 'result/'

def getDeltaSeconds(df_Face, df_Imsi):
    df_FaceCode = pd.merge_asof(df_Face, df_Imsi, on=["TimeStamp"], by=["FaceLabel", "DeviceID"], tolerance=maxDeltaT, direction="nearest")

    df_FaceCode["TimeDelta"] = df_FaceCode.apply(lambda x:pd.Timedelta(seconds=0) if(pd.isna(x['Time1_x']) or pd.isna(x['Time1_y'])) else abs(x['Time1_x'] - x['Time1_y']), axis=1) #无法处理NaN
    df_FaceCode['TimeDelta'] = df_FaceCode['TimeDelta'].fillna(pd.Timedelta(seconds=0))

    df_FaceCode['TimeDeltaSeconds'] = df_FaceCode['TimeDelta'].map(lambda x: x.seconds)

    df_min = df_FaceCode.groupby(["FaceLabel", "DeviceID"])["TimeDeltaSeconds"].min()

    df_describe = df_min.groupby(["DeviceID"]).describe()
    df_describe['edge1'] = df_describe['mean'] + 3 * df_describe['std'] #3倍标准差
    df_describe['edge2'] = df_describe['75%'] + 1.5 * (df_describe['75%'] - df_describe['25%']) # 箱线图四分位确定

    #print(df_describe)
    deltaSeconds = df_describe.apply(lambda x: min(x['edge1'], x['edge2'], x['max']), axis=1)

    return deltaSeconds

path_imsi = train_folder + 'CCF2021_run_record_c_Train.csv'
path_face = train_folder + 'CCF2021_run_record_p_Train.csv'

# 处理表格内的数据
df_Imsi=pd.read_csv(path_imsi,dtype=str)
df_Imsi.columns=['DeviceID','Lon','Lat','Time','Code']
df_Imsi['Time1']=pd.to_datetime(df_Imsi['Time'])
df_Imsi['TimeStamp']=[int(t.timestamp()) for t in df_Imsi['Time1']]
df_Face=pd.read_csv(path_face,dtype=str)
df_Face.columns=['DeviceID','Lon','Lat','Time','FaceLabel']
df_Face['Time1']=pd.to_datetime(df_Face['Time'])
df_Face['TimeStamp']=[int(t.timestamp()) for t in df_Face['Time1']]

path_label = train_folder + 'CCF2021_run_label_Train.csv'
df_label = pd.read_csv(path_label,dtype=str)
dict_label = {}
for tup in zip(df_label['人员编号'], df_label['特征码']):
    dict_label[tup[1]] = tup[0]
# key: 特征码， value: 人员编号

    
df_Face = df_Face.sort_values(by="TimeStamp")
df_Imsi = df_Imsi.sort_values(by="TimeStamp")
df_Imsi["FaceLabel"] = df_Imsi['Code'].map(lambda x: dict_label.get(x))

deltaSeconds = getDeltaSeconds(df_Face, df_Imsi)
#print(deltaSeconds)

df = get_MatchRecord(df_Imsi,df_Face, deltaSeconds)
res = genFeature(df,df_Face,df_Imsi, deltaSeconds)
res['label'] = res.apply(label,axis=1)
res['testflag'] = res.apply(testflag, axis=1)
train = res[res['testflag'] == 0]
test = res[res['testflag'] == 1]

X = res[res.columns[2:-2]]
y = res['label']

X_train = train[train.columns[2:-2]] #First 2 Columns: FaceLabel Code   Last 2 columns: label testflag
y_train = train['label']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
model = XGBClassifier(scale_pos_weight=100, learning_rate=0.05, random_state=1000)
#model.fit(X_train,y_train)
print('== Start Training ==')
model.fit(X,y)
print('== End Training ==')

resX = res[res.columns[2:-2]]
probability = model.predict_proba(resX)[:,1]
res['probability'] = pd.Series(probability)

#xgb_temp = res[res['testflag']==1].groupby("FaceLabel").apply(lambda t: t[t.probability==t.probability.max()].iloc[0])
xgb_temp = res.groupby("FaceLabel").apply(lambda t: t[t.probability==t.probability.max()].iloc[0])

xgb_temp["label"] = xgb_temp.apply(label, axis=1)
precision_xgb = len(xgb_temp[ xgb_temp['label'] == 1 ]) / len(xgb_temp)
print("xgboost计算的正确率为：", len(xgb_temp[ xgb_temp['label'] == 1 ]),  len(xgb_temp), str(precision_xgb))


path_imsi = test_folder + 'CCF2021_run_record_c_EvalA.csv'
path_face = test_folder + 'CCF2021_run_record_p_EvalA.csv'

df_Imsi=pd.read_csv(path_imsi,dtype=str)
df_Imsi.columns=['DeviceID','Lon','Lat','Time','Code']
df_Imsi['Time1']=pd.to_datetime(df_Imsi['Time'])
df_Imsi['TimeStamp']=[int(t.timestamp()) for t in df_Imsi['Time1']]
df_Face=pd.read_csv(path_face,dtype=str)
df_Face.columns=['DeviceID','Lon','Lat','Time','FaceLabel']
df_Face['Time1']=pd.to_datetime(df_Face['Time'])
df_Face['TimeStamp']=[int(t.timestamp()) for t in df_Face['Time1']]

df_Face = df_Face.sort_values(by="TimeStamp")
df_Imsi = df_Imsi.sort_values(by="TimeStamp")


df_FaceCode = pd.merge_asof(df_Imsi, df_Face, on=["TimeStamp"], by=["DeviceID"], tolerance=maxDeltaT, direction="nearest")
df_FaceCode["TimeDelta"] = df_FaceCode.apply(lambda x:pd.Timedelta(seconds=0) if(pd.isna(x['Time1_x']) or pd.isna(x['Time1_y'])) else abs(x['Time1_x'] - x['Time1_y']), axis=1) #无法处理NaN
df_FaceCode['TimeDelta'] = df_FaceCode['TimeDelta'].fillna(pd.Timedelta(seconds=0))

df_FaceCode['TimeDeltaSeconds'] = df_FaceCode['TimeDelta'].map(lambda x: x.seconds)

df_min = df_FaceCode.groupby(["Code", "DeviceID"])["TimeDeltaSeconds"].min()

#df_describe = df_min[df_min > 2].groupby(["DeviceID"]).describe()
df_describe = df_min.groupby(["DeviceID"]).describe()
df_describe['edge1'] = df_describe['mean'] + 3 * df_describe['std'] #3倍标准差

#deltaSeconds = df_describe.apply(lambda x: min(x['edge1'], maxDeltaT), axis=1) #采用3倍标准差方法
deltaSeconds = df_describe.apply(lambda x: min(x['max'], maxDeltaT), axis=1)

#print(df_describe)
#deltaSeconds = {"D00":maxDeltaT, "D01":maxDeltaT, "D02":maxDeltaT, "D03":maxDeltaT, "D04":maxDeltaT, "D05":maxDeltaT}
#print(deltaSeconds)

df = get_MatchRecord(df_Imsi,df_Face, deltaSeconds)

res = genFeature(df,df_Face,df_Imsi, deltaSeconds)

resX = res[res.columns[2:]]
print('== Start Testing ==')
probability = model.predict_proba(resX)[:,1]
print('== End Testing ==')
res['probability'] = pd.Series(probability)

xgb_temp = res.groupby("FaceLabel").apply(lambda t: t[t.probability==t.probability.max()].iloc[0])

path_pred = result_folder + "CCF2021_run_pred_EvalA.csv"
df_pred=pd.DataFrame(zip(xgb_temp['FaceLabel'], xgb_temp['Code']), columns=['人员编号', '特征码']).sort_values(by=['人员编号'])
df_pred.to_csv(path_pred,index=False) 

## lost data : P0433, P0592, P0286, P0304