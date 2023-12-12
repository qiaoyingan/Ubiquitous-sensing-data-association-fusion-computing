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

train_folder, test_folder, result_folder = 'train_dataset/', 'test_dataset/', 'result/'

path_imsi = train_folder + 'CCF2021_run_record_c_Train.csv'
path_face = train_folder + 'CCF2021_run_record_p_Train.csv'

dict_label = {} # dict_label is used to identify the correct person-code relation
device_list = [] # device_list is used to record all the positions (Dxx)
face_list, code_list = [], [] # xxxx_list is used to record the info of different positions
window_size, stride = 100, 1000 # identify person and code appear in [time, time + window_size] as co-appear
co_appear_dict = {} # record person and code appear in the same time block

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
    # co_appear_dict = {'Facelabel' : [], 'Code' : []}

    for _ in range(len(device_list)): # Each position

        start_time = min(face_list[_]['TimeStamp'].iloc[0], code_list[_]['TimeStamp'].iloc[0])
        end_time = start_time + window_size
        the_end = max(face_list[_]['TimeStamp'].iloc[-1], code_list[_]['TimeStamp'].iloc[-1])

        face_len, code_len = len(face_list[_]), len(code_list[_])
        print(face_len, code_len)
        print(" = = = ")
        face_index, code_index = 0, 0
        start_face, start_code = 0, 0 # record the start index in last window

        # There are still dispute:
            # 1、when a particular person/code occur more than once in current window, n_pc++? n_p++? n_c++?
            # 2、when current window have no more person, still count code?
            # 3、when current window have no more code, still count person?
        # Current approach: 1: +=1, 2: No, 3: No.

        # * the algorithm seems to be very slow, especially set window_size & stripe into small values.

        while 1: # slide window [start_time, end_time], nxt window is [start_time + stride, end_time + stride]
            print(face_index, code_index, start_face, start_code, start_time, end_time)
            faces, codes = [], []
            face_index, code_index = start_face, start_code # prevent to start from 0 for saving time

            while 1: # find the persons and codes in current window and add them into faces[], codes[]
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

            a = 0
            for f in faces:
                for c in codes:
                    if (f, c) in co_appear_dict:
                        co_appear_dict[(f, c)] += 1
                    else:
                        co_appear_dict[(f, c)] = 1
            # update the window
            start_time += stride
            end_time += stride
            if start_time > the_end:
                break
    print(co_appear_dict)

    # df_Face1 = df_Face.loc[df_Face['DeviceID'] == 'D02']
    # df_Imsi1 = df_Imsi[['DeviceID', 'TimeStamp', 'Code']]
    # print(face_list[0], code_list[0])
    # # 这个做法是对facelabel中的每个人脸选取一个时空最近的硬件码标记为同时出现，
    # # 后续可以改进的是在某个时间段内出现的所有硬件码都标记为同时出现
    # df = pd.merge_asof(df_Face1, df_Imsi1, on=["TimeStamp"], by=["DeviceID"], tolerance=maxDeltaT,
    #                 direction="nearest")
    # # print(df)
    # df = df[['FaceLabel', 'Code']]
    # print(df)

    # # 创建一个矩阵，按照FaceLabel和code进行分组，统计每个组合出现的次数
    # matrix = pd.pivot_table(df, index="FaceLabel", columns="Code", aggfunc="size", fill_value=0)
    # # 打印矩阵
    # print(matrix)
    # print(len(matrix))
    # print(matrix.columns)
    # print(matrix.loc['P0000'].idxmax())  # 根据行索引选择X行的最大值的列名称
    # print(matrix.iloc[0].idxmax())  # 根据行位置选择第一行的最大值的列名称，注意Python的索引从0开始

    # matrix['Row_sum'] = matrix.apply(lambda x: x.sum(), axis=1)  # 按行求和，添加为新列
    # matrix.loc['Col_sum'] = matrix.apply(lambda x: x.sum())  # 按列求和，添加为新行
    # print(matrix)
    # ma = matrix
    # # print(matrix.loc['P1995', 'C012FmvR'])
    # # for index, row in matrix.iteritems():

    # print(ma.iloc[-1, -1])
    # for in1, (face, row) in enumerate(matrix.iterrows()):
    #     for in2, n_pc in enumerate(row):
    #         ma.iloc[in1, in2] = np.log(1 + n_pc) * np.log(1 + n_pc/(ma.iloc[-1, in2]-n_pc+1))*np.log(1 + n_pc/(ma.iloc[in1, -1]-n_pc+2))
    #         # print(ma.iloc[in1, in2])
    # print(ma)

    # 计算公式 
    # np.log(1+ n_pc)*np.log(1+ n_pc/(n_c-n_pc+1))*np.log(1+ n_pc/(n_p-n_pc+2))

