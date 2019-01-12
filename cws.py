#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:cws.py
@TIME:2018/5/10 17:25
@DES:
'''

# '''这里填写所有的依赖'''
from flask import request,jsonify,Response
from flask_cors import *
from flask import Flask
from flask import current_app
from flask import g

from keras.models import load_model
import re
import numpy as np
import pandas as pd
import codecs
import random


import jieba
import jieba.posseg as pseg
import jieba.analyse

# from cws_support import get_chars,get_datas

app = Flask(__name__)
CORS(app, supports_credentials=True)
ctx=app.app_context()
ctx.push()
# '''设定全局通用变量'''

maxlen = 32  #输入窗口大小
batch_size = 1024
model_file = "web/dev_model.h5"
cost_file = "web/dev_cost_file.txt"

global maxAcc
maxAcc = 0




#模型使用测试
# s = '我会成功'
# s = s.decode('utf-8')
# print cws_cut(s,chars_pku,model_pku04,maxlen)
# print cws_cut(s,chars_pku,model_pku,maxlen)
# print cws_cut(s,chars_msr,model_msr,maxlen)



#模型的测试演示
# evaluate_model01(model_pku,test_pku,maxlen,batch_size)
# evaluate_model01(model_pku04,test_pku,maxlen,batch_size)
# evaluate_model01(model_msr,test_msr,maxlen,batch_size)



# model = gener_model03(maxlen,len(chars_pku),128,64,dev_model)
# model = gener_model02(maxlen,len(chars_pku),128,64,dev_model)
# model = gener_model(maxlen,len(chars_pku),128,64,dev_model)
# model.save(dev_model)


# 接口实现

'''
页面：home  
分词接口实现
'''
@app.route('/cut/jieba',methods=['POST','GET'])
def jieba_cut():
    input_s = request.args.get('input_s', '')
    result = jieba.cut(input_s)
    result = '  '.join(result)
    return jsonify({'result': result})

@app.route('/cut/S', methods=['POST', 'GET'])
def cut_S():
    global graph
    with graph.as_default():
        input_s = request.args.get('input_s', '')
        result = cws_cut(input_s, chars_msr, model_S, maxlen)
        return jsonify({'result': result})

@app.route('/cut/T', methods=['POST', 'GET'])
def cut_T():
    global graph
    with graph.as_default():
        input_s = request.args.get('input_s', '')
        result = cws_cut(input_s, chars_as, model_T, maxlen)
        return jsonify({'result': result})

@app.route('/',methods=['POST','GET'])
def my_cut():
    return "hello, welcome to Sunpart"



'''
简繁体转换
'''
from language import *

@app.route('/cut/toS', methods=['POST', 'GET'])
def toS():
    global graph
    with graph.as_default():
        input_s = request.args.get('input_s', '')
        result = toSimple(input_s)
        return jsonify({'result': result})

@app.route('/cut/toT', methods=['POST', 'GET'])
def toT():
    global graph
    with graph.as_default():
        input_s = request.args.get('input_s', '')
        result = toTrandition(input_s)
        return jsonify({'result': result})


'''
页面：evaluate
'''
import sys

@app.route('/eva', methods=['POST', 'GET'])
def evaluate():
    test_data_name = request.args.get('test_data', '')
    model_name = request.args.get('model', '')
    test_batch_size = request.args.get('batch_size', '')
    print model_name,test_data_name,test_batch_size
    if(test_data_name=="PKU"):
        eva_data = test_msr  #后面需要改正过来
    elif(test_data_name=="MSR"):
        eva_data = test_msr
    elif(test_data_name=="AS"):
        eva_data = test_as
    else:
        msg = "数据集选择错误："+test_data_name
        status = '1'
        return jsonify({'status':status,'msg':msg})
    if(model_name=="pku"):
        eva_model = model_S   #后面需要改正过来
    elif(model_name=='S'):
        eva_model = model_S
    elif(model_name=='T'):
        eva_model = model_T
    else:
        msg = "模型选择错误：" + model_name
        status = '1'
        return jsonify({'status': status, 'msg': msg})
    batch_size = int(test_batch_size)

    # 将终端输出指向文件
    output = sys.stdout
    outputfile = open('web/evaluation.txt', 'w')
    sys.stdout = outputfile

    global graph
    with graph.as_default():
        loss,acc =evaluate_model01(eva_model,eva_data,maxlen,batch_size)

    # 将终端输出修改回来
    outputfile.close()
    sys.stdout = output

    status = '0'
    return jsonify({'status': status,
                    'loss':loss,
                    'acc':acc})

'''develop 页面'''
@app.route('/dev/generate', methods=['POST', 'GET'])
def generate_model():
    generate_way = request.args.get('generate_way', '')
    chars = request.args.get('chars', '')
    word_size = request.args.get('word_size', '')
    num_lstm = request.args.get('num_lstm', '')
    print generate_way, chars, word_size,num_lstm
    if(chars == "PKU"):
        dev_chars =chars_pku
    elif(chars == "MSR"):
        dev_chars = chars_msr
    elif(chars == "AS"):
        dev_chars = chars_as
    else:
        msg = "字典选择错误："+str(chars)
        status = '1'
        return jsonify({'msg':msg,
                        'status':status})
    # 将终端输出指向文件
    output = sys.stdout
    outputfile = open('web/dev_model_info.txt', 'w')
    sys.stdout = outputfile
    print "generate_way  chars  word_size  num_lstm"
    print generate_way,chars,word_size,num_lstm
    global graph
    with graph.as_default():
        if(generate_way == '1'):
            gener_model(maxlen, len(dev_chars), int(word_size),int(num_lstm),model_file)
        elif(generate_way =='2'):
            gener_model02(maxlen, len(dev_chars), int(word_size), int(num_lstm), model_file)
        else:
            gener_model03(maxlen,len(dev_chars),int(word_size),int(num_lstm),model_file)


    status = '0'
    # 将终端输出修改回来
    outputfile.close()
    sys.stdout = output
    # 清空cost文件数据
    f1 = open('web/dev_cost_file.txt', 'w')
    f1.write('')
    f1.close()

    global  maxAcc
    maxAcc = 0

    return jsonify({'status': status})

@app.route('/dev/train', methods=['POST', 'GET'])
def train_model():
    f2 = open('web/dev_model_train.txt', 'w')
    f2.write("正在准备中，请稍等……")
    f2.close()
    train_way = request.args.get('train_way', '')
    chars = request.args.get('chars', '')
    batch_size = request.args.get('batch_size', '')
    epoch = request.args.get('epoch', '')
    if (chars == "PKU"):
        dev_train = test_pku
        dev_test = test_pku
    elif (chars == "MSR"):
        dev_train = test_msr
        dev_test = test_msr
    elif(chars == "AS"):
        dev_test = test_as
        dev_train = test_as
    else:
        msg = "字典选择错误：" + str(chars)
        status = '1'
        return jsonify({'msg': msg,
                        'status': status})

    # print "正在准备……"
    global graph
    with graph.as_default():
        model = get_model(model_file)
        # 将终端输出指向文件
        output = sys.stdout
        outputfile = open('web/dev_model_train.txt', 'w')
        sys.stdout = outputfile
        if(train_way=='0'):
            train_model01_1(model, dev_train, maxlen, int(batch_size), int(epoch), model_file,cost_file)
        elif(train_way == '1'):
            # 只训练不评估
            train_model01(model,dev_train,maxlen,int(batch_size),int(epoch),model_file,cost_file)
        elif(train_way == '2'):
            # test in train
            global maxAcc
            maxAcc=train_model02(model, dev_train, maxlen, int(batch_size), int(epoch), model_file, cost_file,maxAcc)
        else:
            # test out train
            global maxAcc
            maxAcc=train_model03(model, dev_train, dev_test,maxlen, int(batch_size), int(epoch), model_file, cost_file,maxAcc)


    status = '0'
    # 将终端输出修改回来
    outputfile.close()
    sys.stdout = output
    # result = cws_cut(input_s, chars_msr, model_msr, maxlen)
    return jsonify({'status': status})
    # model, train_data, test_data, maxlen, batch_size, epoch, model_file, cost_file

@app.route('/dev/model_para',methods=['POST', 'GET'])
def model_para():
    model_file = open('web/dev_model_info.txt', 'w')
    data = model_file.readlines()
    info = data[1]
    info = info.strip(' ')
    generate_way = info[0]
    chars = info[1]
    word_size =info[2]
    lstm_node = info[3]
    return jsonify({'generate_way':generate_way,
                    'chars':chars,
                    'word_size':word_size,
                    'lstm_node':lstm_node})



import tensorflow as tf



if __name__ == '__main__':

    graph = tf.get_default_graph()
    global graph
    with graph.as_default():
        # '''运行的时候加载数据'''

        # 加载模型

        from functions_pre import *
        # from function_after import *
        from use_model import *

        print "正在加载模型……"
        # model01 = get_model('model/my_model.h5')
        # model02 = get_model('model/my_model_02.h5')
        # 用于分词的模型
        model_T = get_model('model/model_asV2.h5')
        model_S = get_model('model/model_msrV2.h5')

        # 用于测试的模型
        # model_pku = get_model('model/model_pku.h5')
        # model_pku04 = get_model('model/model_pku04.h5')
        # model_msr = get_model('model/model_msr.h5')

        # 加载字典
        print "正在加载字典……"
        # chars01 = get_chars('dictionary/chars.txt')
        chars_pku = get_chars('dictionary/chars02_pku.txt')
        chars_msr = get_chars('dictionary/chars02_msr.txt')
        chars_as = get_chars('dictionary/char04_as.txt')


        # 加载测试集
        print "正在加载测试集……"
        test_msr = init_datas('data_set/test_msr.txt',chars_msr,maxlen)
        test_as = init_datas('data_set/test_as02.txt',chars_as,maxlen)
        test_pku = init_datas('data_set/test_pku.txt',chars_pku,maxlen)


        # 加载训练集

        # print "正在加载训练集……"
        # train_msr = init_datas('data_set/train_msr.txt',chars_msr,maxlen)
        # train_pku = init_datas('data_set/train_pku.txt',chars_pku,maxlen)


        # f2 = open('web/dev_model_train.txt', 'w')
        # f2.write("请先训练模型！")
        # f2.close()

        app.run()