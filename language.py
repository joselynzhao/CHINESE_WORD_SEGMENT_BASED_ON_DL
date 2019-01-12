#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:language.py
@TIME:2018/5/29 20:44
@DES:
'''

from langconv import *
# 转换繁体到简体
def cht_to_chs(line):
     line = Converter('zh-hans').convert(line)
     line.encode('utf-8')
     return line

 # 转换简体到繁体
def chs_to_cht(line):
     line = Converter('zh-hant').convert(line)
     line.encode('utf-8')
     return line


def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

def Simplified2Traditional(sentence):
    '''
    将sentence中的简体字转为繁体字
    :param sentence: 待转换的句子
    :return: 将句子中简体字转换为繁体字之后的句子
    '''
    sentence = Converter('zh-hant').convert(sentence)
    return sentence

def toSimple(sentence):
    oldSentence = sentence
    # oldSentence=oldSentence.decode('utf-8')
    newSentence = Traditional2Simplified(oldSentence)
    return newSentence.encode('utf-8')

def toTrandition(sentence):
    oldSentence = sentence
    # oldSentence = oldSentence.decode('utf-8')
    newSentence = Simplified2Traditional(oldSentence)
    return newSentence.encode('utf-8')

if __name__=="__main__":
    print toSimple('憂郁的臺灣烏龜')
    print toTrandition('忧郁的台湾乌龟')
    # traditional_sentence = '憂郁的臺灣烏龜'
    # traditional_sentence = traditional_sentence.decode('utf-8')
    # simplified_sentence = Traditional2Simplified(traditional_sentence)
    # simplified_sentence=simplified_sentence.encode('utf-8')
    # print(simplified_sentence)

    '''
    输出结果：
        忧郁的台湾乌龟
    '''

