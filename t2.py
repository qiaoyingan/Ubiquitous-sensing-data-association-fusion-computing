# -*- coding: utf-8 -*-
"""
CCF2021 run 算法基准程序
@author: zhouwei
"""

import pandas as pd

import random
import numpy as np

rdseed = 10
random.seed(rdseed)
np.random.seed(rdseed)

num_people = 2000

maxDeltaT = 120  # 120秒为最长采集时差


def getDeltaSeconds(df_Face, df_Imsi):
    df_FaceCode = pd.merge_asof(df_Face, df_Imsi, on=["TimeStamp"], by=["FaceLabel", "DeviceID"], tolerance=maxDeltaT,
                                direction="nearest")

    print(df_FaceCode)
    print(df_FaceCode.columns)
    df_FaceCode["TimeDelta"] = df_FaceCode.apply(
        lambda x: pd.Timedelta(seconds=0) if (pd.isna(x['Time1_x']) or pd.isna(x['Time1_y'])) else abs(
            x['Time1_x'] - x['Time1_y']), axis=1)  # 无法处理NaN
    df_FaceCode['TimeDelta'] = df_FaceCode['TimeDelta'].fillna(pd.Timedelta(seconds=0))

    df_FaceCode['TimeDeltaSeconds'] = df_FaceCode['TimeDelta'].map(lambda x: x.seconds)

    df_min = df_FaceCode.groupby(["FaceLabel", "DeviceID"])["TimeDeltaSeconds"].min()

    df_describe = df_min.groupby(["DeviceID"]).describe()
    df_describe['edge1'] = df_describe['mean'] + 3 * df_describe['std']  # 3倍标准差
    df_describe['edge2'] = df_describe['75%'] + 1.5 * (df_describe['75%'] - df_describe['25%'])  # 箱线图四分位确定

    # print(df_describe)
    deltaSeconds = df_describe.apply(lambda x: min(x['edge1'], x['edge2'], x['max']), axis=1)

    return deltaSeconds


path_imsi = './train_dataset/CCF2021_run_record_c_Train.csv'
path_face = './train_dataset/CCF2021_run_record_p_Train.csv'

df_Imsi = pd.read_csv(path_imsi, dtype=str)
df_Imsi.columns = ['DeviceID', 'Lon', 'Lat', 'Time', 'Code']
df_Imsi['Time1'] = pd.to_datetime(df_Imsi['Time'])
df_Imsi['TimeStamp'] = [int(t.timestamp()) for t in df_Imsi['Time1']]
# print(df_Imsi['TimeStamp'])
df_Face = pd.read_csv(path_face, dtype=str)
df_Face.columns = ['DeviceID', 'Lon', 'Lat', 'Time', 'FaceLabel']
df_Face['Time1'] = pd.to_datetime(df_Face['Time'])
df_Face['TimeStamp'] = [int(t.timestamp()) for t in df_Face['Time1']]
# print(df_Face['TimeStamp'])

path_label = './train_dataset/CCF2021_run_label_Train.csv'
df_label = pd.read_csv(path_label, dtype=str)
dict_label = {}
for tup in zip(df_label['人员编号'], df_label['特征码']):
    dict_label[tup[1]] = tup[0]

# print(len(dict_label))

df_Face = df_Face.sort_values(by="TimeStamp")
df_Imsi = df_Imsi.sort_values(by="TimeStamp")
df_Imsi["FaceLabel"] = df_Imsi['Code'].map(lambda x: dict_label.get(x))
# print(len(df_Face))
# print(len(df_Imsi))
df_Face1 = df_Face[['DeviceID', 'TimeStamp', 'FaceLabel']]
df_Imsi1 = df_Imsi[['DeviceID', 'TimeStamp', 'Code']]
# deltaSeconds = getDeltaSeconds(df_Face1, df_Imsi1)

# print(deltaSeconds)


# 合并两个文件，保留相同的DeviceID和Time
# df = pd.merge(df_Face1, df_Imsi1, on=["DeviceID", "TimeStamp"], how="inner")

# 这个做法是对facelabel中的每个人脸选取一个时空最近的硬件码标记为同时出现，
# 后续可以改进的是在某个时间段内出现的所有硬件码都标记为同时出现
df = pd.merge_asof(df_Face1, df_Imsi1, on=["TimeStamp"], by=["DeviceID"], tolerance=maxDeltaT,
                   direction="nearest")
# print(df)
df = df[['FaceLabel', 'Code']]
print(df)

# 创建一个矩阵，按照FaceLabel和code进行分组，统计每个组合出现的次数
matrix = pd.pivot_table(df, index="FaceLabel", columns="Code", aggfunc="size", fill_value=0)
# 打印矩阵
print(matrix)
print(len(matrix))
print(matrix.columns)
print(matrix.loc['P0000'].idxmax())  # 根据行索引选择X行的最大值的列名称
print(matrix.iloc[0].idxmax())  # 根据行位置选择第一行的最大值的列名称，注意Python的索引从0开始

matrix['Row_sum'] = matrix.apply(lambda x: x.sum(), axis=1)  # 按行求和，添加为新列
matrix.loc['Col_sum'] = matrix.apply(lambda x: x.sum())  # 按列求和，添加为新行
print(matrix)
ma = matrix
# print(matrix.loc['P1995', 'C012FmvR'])
# for index, row in matrix.iteritems():

print(ma.iloc[-1, -1])
for in1, (face, row) in enumerate(matrix.iterrows()):
    for in2, n_pc in enumerate(row):
        ma.iloc[in1, in2] = np.log(1 + n_pc) * np.log(1 + n_pc/(ma.iloc[-1, in2]-n_pc+1))*np.log(1 + n_pc/(ma.iloc[in1, -1]-n_pc+2))
        # print(ma.iloc[in1, in2])
print(ma)

# 计算公式 
# np.log(1+ n_pc)*np.log(1+ n_pc/(n_c-n_pc+1))*np.log(1+ n_pc/(n_p-n_pc+2))

