# -*- coding: utf-8 -*-
"""
CCF2021 run 算法基准程序
@author: zhouwei
"""

import pandas as pd

import random
import numpy as np
from xgboost import XGBClassifier

rdseed = 10
random.seed(rdseed)
np.random.seed(rdseed)


def get_matchdf(device_list, face_list, code_list, window_size, stride):
    co_appear_dict ,dict_Face_window, dict_Imsi_window = {}, {}, {}
    for _ in range(len(device_list)):  # Each position

        start_time = min(face_list[_]['TimeStamp'].iloc[0], code_list[_]['TimeStamp'].iloc[0])
        end_time = start_time + window_size
        the_end = max(face_list[_]['TimeStamp'].iloc[-1], code_list[_]['TimeStamp'].iloc[-1])

        face_len, code_len = len(face_list[_]), len(code_list[_])
        # print(face_len, code_len)
        print("=== Position {} Begin ===".format(device_list[_]))
        face_index, code_index = 0, 0
        start_face, start_code = 0, 0  # record the start index in last window

        # There are still dispute:
        # 1、when a particular person/code occur more than once in current window, n_pc++? n_p++? n_c++?
        # 2、when current window have no more person, still count code?
        # 3、when current window have no more code, still count person?
        # Current approach: 1: +=1, 2: No, 3: No.

        while 1:  # slide window [start_time, end_time], nxt window is [start_time + stride, end_time + stride]
            print(face_index, code_index, start_face, start_code, start_time, end_time)
            faces, codes = [], []
            face_index, code_index = start_face, start_code  # prevent to start from 0 for saving time

            while 1:  # find the persons and codes in current window and add them into faces[], codes[]
                # print(1)
                if face_index < face_len:
                    if face_list[_]['TimeStamp'].iloc[face_index] < start_time:
                        face_index += 1
                        start_face = face_index
                    else:
                        if face_list[_]['TimeStamp'].iloc[face_index] < end_time:
                            faces.append(face_list[_]['FaceLabel'].iloc[face_index])
                            # print("faces append {}".format(face_index))
                            face_index += 1
                if code_index < code_len:
                    if code_list[_]['TimeStamp'].iloc[code_index] < start_time:
                        code_index += 1
                        start_code = code_index
                    else:
                        if code_list[_]['TimeStamp'].iloc[code_index] < end_time:
                            codes.append(code_list[_]['Code'].iloc[code_index])
                            # print("codes append {}".format(code_index))
                            code_index += 1
                if face_index >= face_len or code_index >= code_len:
                    break
                if face_list[_]['TimeStamp'].iloc[face_index] >= end_time \
                        and code_list[_]['TimeStamp'].iloc[code_index] >= end_time:
                    break

            for f in faces:
                for c in codes:
                    co_appear_dict[(f, c)] = 1 if (f, c) not in co_appear_dict else co_appear_dict[(f, c)] + 1
                    dict_Face_window[f] = 1 if f not in dict_Face_window else dict_Face_window[f] + 1
                    dict_Imsi_window[c] = 1 if c not in dict_Imsi_window else dict_Imsi_window[c] + 1
            # update the window
            start_time += stride
            end_time += stride
            if start_time > the_end:
                break
        print("=== Position {} End ===".format(device_list[_]))
    return co_appear_dict, dict_Face_window, dict_Imsi_window


def genFeature(l_TM, face_total, imsi_total, face_window, imsi_window):
    # 人脸的总次数
    # df_T = df_Face.groupby('FaceLabel').count().reset_index()[['FaceLabel', 'DeviceID']]
    # df_T.columns = ['FaceLabel', 'countT']  # 计算每一个facelabel出现的次数
    # facelist1 = df_T['FaceLabel'].tolist()
    # face_count1 = df_T['countT'].tolist()

    # # 特征码的总次数
    # df_M = df_Imsi.groupby('Code').count().reset_index()[['Code', 'DeviceID']]
    # df_M.columns = ['Code', 'countM']
    # codelist1 = df_M['Code'].tolist()
    # code_count1 = df_M['countM'].tolist()

    # 汇总所有特征
    # f : [n_p, n_c, n_pc], f 现在同时用于 score(f) 函数计算得分和 model(x) 中 x 的前三项， 其中 n_p, n_c 在这里取得是整个数据集出现的次数，此处需要修改
    # 正确的修改方案应该是：
    # 1、f 中 n_p, n_c 的计算不变，修改 res 中的计算，使得 res 中 n_p, n_c 的来源从 co_appear_dict 中获得；
    #    此时，model 的输入 X 为 (整个数据集中的人脸出现次数，整个数据集中的编码出现次数，滑动窗口中共现次数，关联得分）
    # 2、f 中 n_p, n_c 计算就按照 co_appear_dict 计算，那么 res 函数无需修改；
    #   此时，model 输入 X 为（滑动窗口人脸出现次数， 滑动窗口编码出现次数，滑动窗口中共现次数，关联得分）
    # 当 stride < window 时，滑动窗口出现次数比整个数据集出现次数多，因为两个window间有重叠
    # 目前结果记录在 readme.md 中

    f = []
    for i in range(len(l_TM)):
        face, code = l_TM[i][0], l_TM[i][1]
        # if facelabel not in face_total:
        #     if code not in :
        #         f.append([l_TM[i][2], 0, 0, 0, 0])
        #     else:
        #         f.append([l_TM[i][2], 0, code_count1[codelist.index(code)], 0, code_count[codelist.index(code)]])
        # else:
        #     if code not in codelist:
        #         f.append(
        #             [l_TM[i][2], face_count1[facelist.index(facelabel)], 0, face_count[facelist.index(facelabel)], 0])
        #     else:
        f.append([l_TM[i][2], face_total[face], imsi_total[code], face_window[face], imsi_window[code]])
    return f


def score(res):
    for i in range(len(res)):
        n_pc = res[i][0]
        n_p = res[i][3]
        n_c = res[i][4]
        if n_pc > n_p or n_pc > n_c:
            con_score = 0
        else:
            con_score = np.log(1 + n_pc) * np.log(1 + n_pc / (n_c - n_pc + 1)) * np.log(1 + n_pc / (n_p - n_pc + 2))
        res[i].append(con_score)
    res1 = [row[0:3] + [row[5]] for row in res]

    return res1


def label1(l_TM):
    label = []
    for i in range(len(l_TM)):
        p = l_TM[i][0]
        c = l_TM[i][1]
        # if(c.endswith(p)):
        if dict_label.get(c) == p:
            label.append(1)
        else:
            label.append(0)
    return label


def label(row):
    p = row['FaceLabel']
    c = row['Code']
    # if(c.endswith(p)):
    if dict_label.get(c) == p:
        return 1
    else:
        return 0


train_folder, test_folder, result_folder = 'train_dataset/', 'test_dataset/', 'result/'

path_imsi = train_folder + 'CCF2021_run_record_c_Train.csv'
path_face = train_folder + 'CCF2021_run_record_p_Train.csv'

dict_label = {}  # dict_label is used to identify the correct person-code relation
device_list = []  # device_list is used to record all the positions (Dxx)
face_list, code_list = [], []  # xxxx_list is used to record the info of different positions
window_size, stride = 100, 15  # identify person and code appear in [time, time + window_size] as co-appear
co_appear_dict = None
dict_Face_total, dict_Imsi_total = None, None
dict_Face_window, dict_Imsi_window = None, None

if __name__ == '__main__':

    df_Imsi = pd.read_csv(path_imsi, dtype=str)
    df_Imsi.columns = ['DeviceID', 'Lon', 'Lat', 'Time', 'Code']
    df_Imsi['Time1'] = pd.to_datetime(df_Imsi['Time'])
    df_Imsi['TimeStamp'] = [int(t.timestamp()) for t in df_Imsi['Time1']]
    df_Face = pd.read_csv(path_face, dtype=str)
    df_Face.columns = ['DeviceID', 'Lon', 'Lat', 'Time', 'FaceLabel']
    df_Face['Time1'] = pd.to_datetime(df_Face['Time'])
    df_Face['TimeStamp'] = [int(t.timestamp()) for t in df_Face['Time1']]

    path_label = train_folder + 'CCF2021_run_label_Train.csv'
    df_label = pd.read_csv(path_label, dtype=str)
    for tup in zip(df_label['人员编号'], df_label['特征码']):
        dict_label[tup[1]] = tup[0]

    df_Face = df_Face.sort_values(by="TimeStamp")
    df_Imsi = df_Imsi.sort_values(by="TimeStamp")
    df_Imsi["FaceLabel"] = df_Imsi['Code'].map(lambda x: dict_label.get(x))

    device_list = df_Face['DeviceID'].drop_duplicates().values.tolist()

    for device in device_list:
        face_list.append(df_Face.loc[df_Face['DeviceID'] == device])
        code_list.append(df_Imsi.loc[df_Imsi['DeviceID'] == device])

    dict_Face_total = df_Face.groupby('FaceLabel').count().to_dict()['DeviceID']
    dict_Imsi_total = df_Imsi.groupby('Code').count().to_dict()['DeviceID']

    co_appear_dict, dict_Face_window, dict_Imsi_window = get_matchdf(device_list, face_list, code_list, window_size, stride)
    l = len(co_appear_dict)
    co_appear_l = []
    for k, v in co_appear_dict.items():
        co_appear_l.append([k[0], k[1], v])
    
    print("=== Start Generating Feature Matrix ===")
    matrix = genFeature(co_appear_l, dict_Face_total, dict_Imsi_total, dict_Face_window, dict_Imsi_window)  # 生成关联矩阵
    print("=== End Generating Feature Matrix ===")
    X = score(matrix)  # 计算关联分数
    y = label1(co_appear_l)  # 计算标签
    X = np.array(X)
    y = np.array(y)

    # 用XGBC分类器进行分类
    model = XGBClassifier(scale_pos_weight=100, learning_rate=0.05, random_state=1000)
    print('=== Start Training ===')
    print(X, y)
    model.fit(X, y)  # 模型训练
    print('=== End Training ===')

    probability = model.predict_proba(X)[:, 1]
    res = pd.DataFrame(co_appear_l, columns=['FaceLabel', 'Code', 'Co_appear'])
    res['probability'] = pd.Series(probability.tolist())

    xgb_temp = res.groupby("FaceLabel").apply(lambda t: t[t.probability == t.probability.max()].iloc[0])
    xgb_temp['label'] = xgb_temp.apply(label, axis=1)
    precision_xgb = len(xgb_temp[xgb_temp['label'] == 1]) / len(xgb_temp)
    print("xgboost计算的正确率为：", len(xgb_temp[xgb_temp['label'] == 1]), len(xgb_temp), str(precision_xgb))

    # 测试
    # face_list, code_list = [], []
    # path_imsi = test_folder + 'CCF2021_run_record_c_EvalA.csv'
    # path_face = test_folder + 'CCF2021_run_record_p_EvalA.csv'
    #
    # df_Imsi = pd.read_csv(path_imsi, dtype=str)
    # df_Imsi.columns = ['DeviceID', 'Lon', 'Lat', 'Time', 'Code']
    # df_Imsi['Time1'] = pd.to_datetime(df_Imsi['Time'])
    # df_Imsi['TimeStamp'] = [int(t.timestamp()) for t in df_Imsi['Time1']]
    # df_Face = pd.read_csv(path_face, dtype=str)
    # df_Face.columns = ['DeviceID', 'Lon', 'Lat', 'Time', 'FaceLabel']
    # df_Face['Time1'] = pd.to_datetime(df_Face['Time'])
    # df_Face['TimeStamp'] = [int(t.timestamp()) for t in df_Face['Time1']]
    #
    # df_Face = df_Face.sort_values(by="TimeStamp")
    # df_Imsi = df_Imsi.sort_values(by="TimeStamp")
    # df_Imsi["FaceLabel"] = df_Imsi['Code'].map(lambda x: dict_label.get(x))
    # device_list = df_Face['DeviceID'].drop_duplicates().values.tolist()
    #
    # for device in device_list:
    #     face_list.append(df_Face.loc[df_Face['DeviceID'] == device])
    #     code_list.append(df_Imsi.loc[df_Imsi['DeviceID'] == device])
    #
    # co_appear_dict = get_matchdf(device_list, face_list, code_list, window_size, stride)
    # l = len(co_appear_dict)
    # co_appear_l = []
    # i = 0
    # for k, v in co_appear_dict.items():
    #     co_appear_l.append([k[0], k[1], v])
    #     i += 1
    # matrix = genFeature(co_appear_l, df_Face, df_Imsi)  # 生成关联矩阵
    # X = score(matrix)  # 计算关联分数
    # X = np.array(X)
    #
    # print('== Start Testing ==')
    # probability = model.predict_proba(X)[:, 1]
    # print('== End Testing ==')
    # res = pd.DataFrame(co_appear_l, columns=['FaceLabel', 'Code', 'Co_appear'])
    # res['probability'] = pd.Series(probability.tolist())
    #
    # xgb_temp = res.groupby("FaceLabel").apply(lambda t: t[t.probability == t.probability.max()].iloc[0])
    #
    # path_pred = result_folder + "CCF2021_run_pred_EvalA.csv"
    # df_pred = pd.DataFrame(zip(xgb_temp['FaceLabel'], xgb_temp['Code']), columns=['人员编号', '特征码']).sort_values(
    #     by=['人员编号'])
    # df_pred.to_csv(path_pred, index=False)
