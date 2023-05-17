# -*- coding: utf-8 -*-
# @Author  : twd
# @FileName: train.py
# @Software: PyCharm


import os
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

import numpy as np
np.random.seed(101)
from pathlib import Path



def catch(data, data2, label):
    # preprocessing label and data
    l = len(data)
    chongfu = 0
    for i in range(l):
        ll = len(data)
        idx = []
        each = data[i]
        j = i + 1
        bo = False
        while j < ll:
            if (data[j] == each).all():
                label[i] += label[j]
                idx.append(j)
                bo = True
            j += 1
        t = [i] + idx
        if bo:
            print(t)
            chongfu += 1
            print(data[t[0]])
            print(data[t[1]])
        data = np.delete(data, idx, axis=0)
        data2 = np.delete(data2, idx, axis=0)
        label = np.delete(label, idx, axis=0)

        if i == len(data)-1:
            break
    print('total number of the same data: ', chongfu)

    return data, data2, label



from model import base, BiGRU_base, Transformer_base,CNNTransformer,UpCNNBiGRU,DiladCNNBiGRU,MKey_DiladCNNBiGRU,MKey_DiladCNNBiGRU_two32,MKey_UnetBiGRU,MKey_CNNBiGRUAttention,MKey_CNNBiGRUMultiAttention,MKey_CNNBiLSTMMAttention,DCNN_BiLSTM,Key_Only,CNN_BiLSTM_Key,DCNN222_BiLSTM_Key,DCNN444_BiLSTM_Key,DCNN888_BiLSTM_Key,DCNN248_Key,DCNN248_GRU_Key,DCNN248_LSTM_Key

def train_my(train, para, model_num, model_path):

    Path(model_path).mkdir(exist_ok=True)
    # =============================================
    # data get
    X_train, K_train, y_train = train[0], train[1], train[2]

    # data and label preprocessing
    y_train = keras.utils.to_categorical(y_train)
    X_train, K_train, y_train = catch(X_train, K_train, y_train)
    y_train[y_train > 1] = 1

    # disorganize
    index = np.arange(len(y_train))
    np.random.shuffle(index)
    X_train = X_train[index]
    K_train = K_train[index]
    y_train = y_train[index]

    # train
    length = X_train.shape[1]
    out_length = y_train.shape[1]
    # =================================================
    t_data = time.localtime(time.time())
    with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
        f.write('data process finished: {}m {}d {}h {}m {}s\n'.format(t_data.tm_mon,t_data.tm_mday,t_data.tm_hour,t_data.tm_min,t_data.tm_sec))


    for counter in range(1, model_num+1):
        # get my neural network model
        if model_path == 'base2':
            model = base(length, out_length, para)
        elif model_path == 'BiGRU_base2':
            model = BiGRU_base(length, out_length, para)
        elif model_path == 'Transformer_base':
            model = Transformer_base(length, out_length, para)
        elif model_path == 'CNNTransformer':
            model = CNNTransformer(length, out_length, para)
        elif model_path == 'UpCNNBiGRU':
            model = UpCNNBiGRU(length, out_length, para)
        elif model_path == 'DiladCNNBiGRU':
            model = DiladCNNBiGRU(length, out_length, para)
        elif model_path == 'MKey_DiladCNNBiGRU':
            model = MKey_DiladCNNBiGRU(length, out_length, para)
        elif model_path == 'MKey_DiladCNNBiGRU_two32':
            model = MKey_DiladCNNBiGRU_two32(length, out_length, para)
        elif model_path == 'MKey_UnetBiGRU':
            model = MKey_UnetBiGRU(length, out_length, para)
        elif model_path == 'MKey_CNNBiGRUAttention':
            model = MKey_CNNBiGRUAttention(length, out_length, para)
        elif model_path == 'MKey_CNNBiGRUMultiAttention':
            model = MKey_CNNBiGRUMultiAttention(length, out_length, para)
        elif model_path == 'MKey_CNNBiLSTMMAttention':
            model = MKey_CNNBiLSTMMAttention(length, out_length, para)
        elif model_path == 'DCNN_BiLSTM':
            model = DCNN_BiLSTM(length, out_length, para)
        elif model_path == 'Key_Only':
            model = Key_Only(166, out_length, para)
        elif model_path == 'CNN_BiLSTM_Key':
            model = CNN_BiLSTM_Key(length, out_length, para)
        elif model_path == 'DCNN222_BiLSTM_Key':
            model = DCNN222_BiLSTM_Key(length, out_length, para)
        elif model_path == 'DCNN444_BiLSTM_Key':
            model = DCNN444_BiLSTM_Key(length, out_length, para)
        elif model_path == 'DCNN888_BiLSTM_Key':
            model = DCNN888_BiLSTM_Key(length, out_length, para)
        elif model_path == 'DCNN248_Key':
            model = DCNN248_Key(length, out_length, para)
        elif model_path == 'DCNN248_GRU_Key':
            model = DCNN248_GRU_Key(length, out_length, para)
        elif model_path == 'DCNN248_LSTM_Key':
            model = DCNN248_LSTM_Key(length, out_length, para)
        else:
            print('no model')


        model.fit([X_train, K_train], y_train, nb_epoch=50, batch_size=64, verbose=2)
        # model.fit(X_train, y_train, nb_epoch=50, batch_size=64, verbose=2)
        each_model = os.path.join(model_path, 'model' + str(counter) + '.h5')
        model.save(each_model)

        tt = time.localtime(time.time())
        with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
            f.write('count{}: {}m {}d {}h {}m {}s\n'.format(str(counter),tt.tm_mon,tt.tm_mday,tt.tm_hour,tt.tm_min,tt.tm_sec))



import time
from test import test_my
def train_main(train, test, model_num, dir):

    # parameters
    ed = 100
    ps = 5
    fd = 128
    dp = 0.5
    lr = 0.001
    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}

    train_my(train, para, model_num, dir)

    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('test start time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))

    test_my(test, para, model_num, dir)

    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('test finish time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))

def test_main(train, test, model_num, dir):

    # parameters
    ed = 100
    ps = 5
    fd = 128
    dp = 0.5
    lr = 0.001
    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}



    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('test start time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))

    test_my(test, para, model_num, dir)

    tt = time.localtime(time.time())
    with open(os.path.join(dir, 'time.txt'), 'a+') as f:
        f.write('test finish time: {}m {}d {}h {}m {}s\n'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
