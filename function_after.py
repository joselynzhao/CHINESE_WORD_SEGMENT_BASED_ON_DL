#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:function_after.py
@TIME:2018/5/31 11:18
@DES:
'''

#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn
@CONTACT:zhaojing17@foxmail.com
@SOFTWERE:PyCharm
@FILE:use_model.py
@TIME:2018/5/11 22:36
@DES:
'''

import random

'''
dec:内部函数不供外部调用
'''
def clean_notag(s):  # 整理一下数据，有些不规范的地方
    if u'“' not in s:
        return s.replace(u'”', '')
    elif u'”' not in s:
        return s.replace(u'“', '')
    elif u'‘' not in s:
        return s.replace(u'’', '')
    elif u'’' not in s:
        return s.replace(u'‘', '')
    else:
        return s

def viterbi(nodes):
    zy = {'be': 0.5,
          'bm': 0.5,
          'eb': 0.5,
          'es': 0.5,
          'me': 0.5,
          'mm': 0.5,
          'sb': 0.5,
          'ss': 0.5
          }
    zy = {i: np.log(zy[i]) for i in zy.keys()}
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}
    for l in range(1, len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1] + i in zy.keys():
                    nows[j + i] = paths_[j] + nodes[l][i] + zy[j[-1] + i]
            k = np.argmax(nows.values())
            paths[nows.keys()[k]] = nows.values()[k]
    return paths.keys()[np.argmax(paths.values())]


def simple_cut(s,chars,model,maxlen):
        print len(s)
        if s:
            for word in s:
                if word in chars.index:
                    continue
                else:
                    chars[word] = random.randint(1, len(chars))
                    print chars[word]

            r = \
                model.predict(np.array([list(chars[list(s)].fillna(0).astype(int)) + [0] * (maxlen - len(s))]),
                              verbose=False)[0][:len(s)]
            '''fillna(0).用0代替缺失值'''
            '''astype(int)转换类型为int'''
            print r
            r = np.log(r)
            nodes = [dict(zip(['s', 'b', 'm', 'e'], i[:4])) for i in r]
            t = viterbi(nodes)
            print "t-----------------------------t"
            print t
            words = []
            for i in range(len(s)):
                if t[i] in ['s', 'b']:
                    '''这是两个起头方式'''
                    words.append(s[i])
                else:
                    words[-1] += s[i]
            return words
        else:
            return []

def cut_word(s,not_cuts,chars,model,maxlen):
        result = []
        j = 0
        for i in not_cuts.finditer(s):
            if(len(s[j:i.start()])>maxlen):
                result.extend(simple_cut(s[j:maxlen],chars,model,maxlen))
                result.extend(simple_cut(s[maxlen:i.start()],chars,model,maxlen))
            else:
                result.extend(simple_cut(s[j:i.start()],chars,model,maxlen))
            result.append(s[i.start():i.end()])
            # print s[i.start():i.end()]
            # 直接把数字或者字符添加进去
            j = i.end()
        result.extend(simple_cut(s[j:],chars,model,maxlen))
        # 添加最后的一段
        return "  ".join(result)

'''内部函数不供外部调用'''

def cws_cut(s,chars,model,maxlen):
    not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')
    result = cut_word(s,not_cuts,chars,model,maxlen)
    result = result.encode('utf-8')
    return result

# '''方法暂时不可用'''
# def changeform(s,chars,maxlen):
#     #s已经是unicode格式了
#     s = s.split('\r\n')
#     s = u''.join(map(clean_notag, s))
#     s = re.split(u'[，。！？、]', s)
#     data = []  # 生成训练样本
#     # for line in s:
#     #     data.append(list(line))
#     #
#     for i in s:
#         x = list(list(i))
#         if x:
#             data.append(x[0])
#
#
#     d = pd.DataFrame(index=range(len(data)))
#     d['data'] = data
#     print d['data']
#     d = d[d['data'].apply(len) <= maxlen]
#     d.index = range(len(d))
#     d['x'] = d['data'].apply(lambda x: np.array(list(chars[x]) + [0] * (maxlen - len(x))))
#     return d


from functions_pre import *
if __name__ =='__main__':
    model = get_model('models/model_pku.h5')
    chars = get_chars('chars/chars02_pku.txt')
    maxlen = 32
    while (1):
        s = raw_input("test_content(e to exit)：")
        if (s == 'e'):
            break
        else:
            s = s.decode('utf-8')
            result = cws_cut(s,chars,model,maxlen)
            print result












