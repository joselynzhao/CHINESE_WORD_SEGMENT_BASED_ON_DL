#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:functions_pre.py
@TIME:2018/5/10 22:34
@DES:
'''

import codecs
import  chardet
import  re
import numpy as np
import  pandas as pd
from keras.utils import np_utils

from keras import  Sequential
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional,Dropout
from keras.models import Model
from keras.models import load_model



# global maxAcc

'''私有函数，不供外部调用'''
def clean(s):  # 整理一下数据，有些不规范的地方
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

def get_xy(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:, 0]), list(s[:, 1])

'''私有函数，不供外部调用'''



# 收集字符集
'''
输入：用以收集字符的文件。。是【经过】tag处理的文件。输入文件类型是utf-8
输出：包含所有的字符集的文件。。输出文件类型也是用utf-8 

'''
def collect_chars(inputfile,chars_in_file,chars_out_file):
    chars_in = codecs.open(chars_in_file, 'r', 'utf-8')
    chars_out = codecs.open(chars_out_file,'w','utf-8')
    chars = chars_in.read().split(',')
    length =  len(chars)

    s = open(inputfile).read().decode('utf-8')
    # 输入的文件用utf-8来处理
    s = s.split('\r\n')
    s = u''.join(map(clean, s))
    s = re.split(u'[，。！？、]/[bems]', s)

    data = []  # 生成训练样本

    for i in s:
        x = get_xy(i)
        if x:
            data.append(x[0])

    for i in data:
        chars.extend(i)

    # chars = pd.Series(chars).value_counts()
    # # chars[:] = range(1, len(chars) + 1)
    # for index in chars.index:
    #     print index
    #     chars_out.write(index)

    for char in chars:
        print char
        chars_out.write(char)

    # 关闭所有文件
    chars_in.close()
    chars_out.close()

# 加载数据
'''
输入：指定的【经过】tag处理的文件、完成的字符集【chars08.txt】、【maxlan】
返回：dataFrame格式的数据
'''
def init_datas(data_file,chars,maxlen):
    s = open(data_file).read().decode('utf-8')
    # chars_in = codecs.open(chars_file, 'r', 'utf-8')
    s = s.split('\r\n')
    s = u''.join(map(clean, s))
    s = re.split(u'[，。！？、]/[bems]', s)
    data = []  # 生成训练样本
    label = []  #生成训练标签
    for i in s:
        x = get_xy(i)
        if x:
            data.append(x[0])
            label.append(x[1])

    d = pd.DataFrame(index=range(len(data)))
    d['data'] = data
    d['label'] = label
    d = d[d['data'].apply(len) <= maxlen]
    d.index = range(len(d))
    tag = pd.Series({u's': 0, u'b': 1, u'm': 2, u'e': 3, u'x': 4})

    # # 获取chars
    # chars = open(chars_file).read().decode('utf-8')
    # chars_list = list(chars)
    #
    # chars = pd.Series(chars_list).value_counts()
    # # 按道理说是可以不用这一步的
    # chars[:] = range(1, len(chars) + 1)
    # chars = get_chars(chars_file)

    d['x'] = d['data'].apply(lambda x: np.array(list(chars[x]) + [0] * (maxlen - len(x))))

    def trans_one(x):
        _ = map(lambda y: np_utils.to_categorical(y, 5), tag[x].reshape((-1, 1)))
        _ = list(_)
        _.extend([np.array([[0, 0, 0, 0, 1]])] * (maxlen - len(x)))
        return np.array(_)

    d['y'] = d['label'].apply(trans_one)
    return d


    # return data

# 获取chars的pd列表
'''
输入：字符集文件【chars08.txt】
输出：pd格式的chars
'''
def get_chars(chars_file):
    # 获取chars
    chars = open(chars_file).read().decode('utf-8')  #chars必须要是unnicode
    # chars = open(chars_file).read()
    chars_list = list(chars)
    # chars = pd.Series(chars_list)
    chars = pd.Series(chars_list).value_counts()
    # 按道理说是可以不用这一步的
    chars[:] = range(1, len(chars) + 1)
    return chars


# 生成模型
'''
输入：【maxlen】、完成字符集的长度【len_chars】、字向量长度【word_size】、需要保持模型的文件名称
返回：生成的模型
'''
def gener_model(maxlen,len_chars,word_size,num_lstm,model_file_name):
    model = Sequential()
    model.add(Embedding(len_chars + 1, word_size, input_length=maxlen))
    '''len_chars+1是输入维度，word_size是输出维度，input_length是节点数'''
    model.add(Bidirectional(LSTM(num_lstm, return_sequences=True), merge_mode='sum'))
    model.add(TimeDistributed(Dense(5, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save(model_file_name)
    model.summary()
    return model


def gener_model02(maxlen,len_chars,word_size,num_lstm,model_file_name):
    model = Sequential()
    model.add(Embedding(len_chars + 1, word_size, input_length=maxlen))
    '''len_chars+1是输入维度，word_size是输出维度，input_length是节点数'''
    model.add(LSTM(num_lstm, return_sequences=True))
    model.add(LSTM(num_lstm, return_sequences=True))
    # model.add(Dropout(0.3))
    # model.add(Dense(5,activation='softmax'))
    model.add(TimeDistributed(Dense(5, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save(model_file_name)
    model.summary()
    return model

def gener_model03(maxlen,len_chars,word_size,num_lstm,model_file_name):
    model = Sequential()
    model.add(Embedding(len_chars + 1, word_size, input_length=maxlen))
    '''len_chars+1是输入维度，word_size是输出维度，input_length是节点数'''
    model.add(Bidirectional(LSTM(num_lstm, return_sequences=True), merge_mode='sum'))
    # model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(num_lstm, return_sequences=True), merge_mode='sum'))
    model.add(TimeDistributed(Dense(5, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save(model_file_name)
    model.summary()
    return model


# 加载模型
'''
输入：指定需要加载的model的文件
返回：model
'''
def get_model(model_file_name):
    try:
        import h5py

        # print ('import fine')
    except ImportError:
        h5py = None

    model = load_model(model_file_name)
    print str(model_file_name)+" 加载成功~"
    # model.summary()
    return model


# 训练模型
'''
dec：采用训练一轮测试一轮的方式。
'''
def train_model01(model,train_data,maxlen,batch_size,epoch,model_file,cost_file):
    print("开始训练！")
    cost_f = codecs.open(cost_file,'a','utf-8')
    for i in range(epoch): #训练epoch轮
        print '第'+str(i+1)+"轮训练开始"
        history = model.fit(np.array(list(train_data['x'])), np.array(list(train_data['y'])).reshape((-1, maxlen, 5)),
                            batch_size=batch_size)
        print history.history
        model.save(model_file)

        # 将新的测试数据写进cost_file中
        cost_f.write(str(i)+' : ')
        cost_f.write(str(history.history)+' ;\n')
    cost_f.close()

def train_model01_1(model,train_data,maxlen,batch_size,epoch,model_file,cost_file):
    print("开始训练！")
    cost_f = codecs.open(cost_file,'a','utf-8')
    # for i in range(epoch): #训练epoch轮
    #     print '第'+str(i+1)+"轮训练开始"
    history = model.fit(np.array(list(train_data['x'])), np.array(list(train_data['y'])).reshape((-1, maxlen, 5)),
                            batch_size=batch_size,epochs=epoch)
    print history.history
    model.save(model_file)

    # 将新的测试数据写进cost_file中
    cost_f.write(str(epoch)+' : ')
    s_all = str(history.history).split("],")
    for s in s_all:
        cost_f.write(s+'];\n')
    cost_f.close()

def train_model03(model,train_data,test_data,maxlen,batch_size,epoch,model_file,cost_file,maxAcc):
    print("开始训练！")
    # maxAcc = 0
    cost_f = codecs.open(cost_file,'a','utf-8')
    for i in range(epoch): #训练epoch轮
        print '第'+str(i+1)+"轮训练开始"
        history = model.fit(np.array(list(train_data['x'])), np.array(list(train_data['y'])).reshape((-1, maxlen, 5)),
                            batch_size=batch_size,validation_data=(
                np.array(list(test_data['x'])), np.array(list(test_data['y'])).reshape((-1, maxlen, 5))))
        s = str(history.history)
        print s
        s = re.findall('\[(.*?)\]', s)
        val_acc = s[2]
        # global maxAcc
        if (val_acc > maxAcc):
            maxAcc = val_acc
            model.save(model_file)
            print "maxAcc = "+str(maxAcc)
            print "------------------------保存模型-----------------------------"
            cost_f.write(' * ')
         #当模型的acc有一定的提升时就保存模型

        # 将新的测试数据写进cost_file中
        cost_f.write(str(i)+' : ')
        cost_f.write(str(history.history) + ' ;\n')
    cost_f.close()
    return maxAcc

'''
dec：采用自我评估的方式,validation_split=0.1
'''
def train_model02(model,train_data,maxlen,batch_size,epoch,model_file,cost_file,maxAcc):
    print("开始训练！")
    # maxAcc = 0
    cost_f = codecs.open(cost_file, 'a', 'utf-8')
    for i in range(epoch): #训练epoch轮
        print '第'+str(i+1)+"轮训练开始"
        history = model.fit(np.array(list(train_data['x'])), np.array(list(train_data['y'])).reshape((-1, maxlen, 5)),
                            batch_size=batch_size,validation_split=0.1)
        s = str(history.history)
        print s
        s = re.findall('\[(.*?)\]', s)
        val_acc = s[2]
        # global maxAcc
        if(val_acc>maxAcc):
            maxAcc = val_acc
            model.save(model_file)
            cost_f.write(' * ')
            print "maxAcc = " + str(maxAcc)
            print "------------------------保存模型-----------------------------"
        cost_f.write(str(i) + ' : ')
        cost_f.write(str(history.history) + ' ;\n')
    cost_f.close()
    return maxAcc


# 评估模型
'''
dec：返回和输出评估结果
'''
def evaluate_model01(model,test_data,maxlen,batch_size):
    print("开始评估！")
    loss_and_accuracy = model.evaluate(np.array(list(test_data['x'])),
                                       np.array(list(test_data['y'])).reshape((-1, maxlen, 5)),
                                       batch_size=batch_size)
    print loss_and_accuracy
    return loss_and_accuracy


import codecs

def character_tagging(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word) == 1:
                output_data.write(word + "/s  ")
            else:
                output_data.write(word[0] + "/b  ")
                for w in word[1:len(word)-1]:
                    output_data.write(w + "/m  ")
                output_data.write(word[len(word)-1] + "/e  ")
            output_data.write("\n")
    input_data.close()
    output_data.close()


