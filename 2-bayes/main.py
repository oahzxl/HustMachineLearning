import os
import jieba
import numpy
import math

def readtxt(path,encoding):                                             #用于读取txt文件
    with open(path, 'r', encoding = encoding) as f:
        lines = f.readlines()
    return lines

def fileWalker(path):
    fileArray = []
    for root, dirs, files in os.walk(path):
       for fn in files:
           eachpath = str(root+'\\'+fn)
           fileArray.append(eachpath)
    return fileArray

def get_word(filename):
    word_list = []
    word_set = []
    file_paths = fileWalker(filename)
    for file_path in file_paths:
      word = file_paths(file_path)
      word_list.append(word)
      word_set.extend(word)
      return word_list, set(word_set)

def email_parser(email_path,invalid_word):                               #得到邮件里的列表
    content_list = readtxt(email_path, 'gbk')
    temp = []
    skip = 0
    for i in content_list:
        if((i != '\n')&(not skip)):
            continue
        if(skip):
            temp.append(i)
        skip = 1
    content_list = temp
    content = (' '.join(content_list)).replace('\n', ' ').replace('\t', ' ').replace(' ', '').replace(' ', '')
    seg_list = jieba.cut(content)
    clean_word = []
    for seg in seg_list:
        if seg not in invalid_word:
        #for w in invalid_word:
        #    if (w not in seg):
                clean_word.append(seg)
    return clean_word

testfile = {}
label = {'s':0,'h':0}
word = {'s':{},'h':{}}
wordnum = {'s':0,'h':0}
invalid_list = readtxt('.\\data\\中文停用词表.txt','utf8')
invalid = (' '.join(invalid_list)).replace(' ','').replace(' ','').split('\n')     #构造停用词表
index = readtxt('.\\data\\newindex','utf8')
filelist = (' '.join(index)).replace(' ','').replace(' ','').split('\n')           #读取newindex中的信息
testnum = int(len(filelist)*0.3)                                                   #test与train比例为3：7
testrank = numpy.random.choice(len(filelist),size=testnum,replace=False)           #随机选取测试集
testfilelist = []
tnum = 0
fnum = 0
errorset = []
accuracy = 0.0
for r in testrank:
    testfilelist.append(filelist[r])                                               #testfilelist作为测试集
for r in testfilelist:
    filelist.remove(r)                                                             #filelist作为训练集
for f in filelist:                                                                 #记录训练集信息
    label[f[0]] = label[f[0]] + 1
    tempadd = '.\\data\\email\\trec06c\\data\\' + f[-7:-4] + '\\' + f[-3:]
    tempword = email_parser(tempadd,invalid)
    for w in tempword:
        wordnum[f[0]] = wordnum[f[0]] + 1
        if(w in word[f[0]]):
            (word[f[0]])[w] = (word[f[0]])[w] + 1
        else:
            (word[f[0]])[w] = 1
for f in testfilelist:                                                             #记录test中分词的信息
    tempadd = '.\\data\\email\\trec06c\\data\\' + f[-7:-4] + '\\' + f[-3:]
    tempword = email_parser(tempadd,invalid)
    tempdict = {}
    for w in tempword:
        if(w in tempdict):
            tempdict[w] = tempdict[w] + 1
        else:
            tempdict[w] = 1
    testfile[tempadd] = [f[0],tempdict]
wordset = set([])                                                                 #准备建立词表
for (k,v) in word['s'].items():                                                   #筛选分词
    if v > 40:
        if (k not in word['h']) or ((word['h'])[k] < 21) :
            wordset.add(k)
for (k,v) in word['h'].items():
    if v > 20:
        if (k not in word['s']) or ((word['s'])[k] < 41) :
            wordset.add(k)
l = 1.0
w = len(wordset)
lw = l*w
P_y = {'h':0.0,'s':0.0}
P_y['h'] = label['h']/(label['h']+label['s'])                                    #计算标签概率
P_y['s'] = label['s']/(label['h']+label['s'])
P_y['h'] = math.log(P_y['h'])
P_y['s'] = math.log(P_y['s'])
P_xh = {}

P_xs = {}
for w in wordset:                                                                #根据训练集计算出条件概率
    if w in word['h'].keys():
        P_xh[w] = ((word['h'])[w] + l)/ (wordnum['h'] + lw)
    else:
        P_xh[w] = l/(wordnum['h'] + lw)
    if w in word['s'].keys():
        P_xs[w] = ((word['s'])[w] + l)/ (wordnum['s'] + lw)
    else:
        P_xs[w] = l/ (wordnum['s'] + lw)
for w in wordset:                                                                #对于符合条件的概率进行放大，至于条件具体是什么报告里可以看到
    if (w not in word['h'].keys()) or ((word['h'])[w] < 3):
        P_xs[w] = 2.0*P_xs[w]
    if (w not in word['s'].keys()) or ((word['s'])[w] < 5):
        P_xh[w] = 2.0*P_xh[w]
for w in wordset:                                                                #根据训练集计算出条件概率
    if w in word['h'].keys():
        P_xh[w] = math.log(P_xh[w])
    else:
        P_xh[w] = math.log(P_xh[w])
    if w in word['s'].keys():
        P_xs[w] = math.log(P_xs[w])
    else:
        P_xs[w] = math.log(P_xs[w])
for (k,v) in testfile.items():                                                   #开始对test进行预测
    ps = P_y['s']
    ph = P_y['h']
    for (w,n) in v[1].items():
        if w in wordset:
            ph = ph + (P_xh[w]*n)
            ps = ps + (P_xs[w]*n)
    if ps > ph:                                                                  #统计准确率
        if v[0] == 's':
            tnum = tnum + 1
        else:
            fnum = fnum + 1
            errorset.append(k)
    elif ps < ph:
        if v[0] == 's':
            fnum = fnum + 1
            errorset.append(k)
        else:
            tnum = tnum + 1
    else:
        if v[0] == 's':
            tnum = tnum + 1
        else:
            fnum = fnum + 1
            errorset.append(k)
accuracy = tnum/testnum

for w in wordset:                                                                    #这往下都是我自己写代码过程中需要打出来看的信息可以忽略
    if not P_xh[w]:
        P_xh.pop(w)
    if not P_xs[w]:
        P_xs.pop(w)
print(accuracy)
print(errorset)
P_xs_max = sorted(P_xs.items(),key = lambda item:item[1],reverse=True)
P_xh_max = sorted(P_xh.items(),key = lambda item:item[1],reverse=True)
print(P_xs_max)
print(P_xh_max)
