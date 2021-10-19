# -*- coding: cp936 -*-
'''
单词总量44084
训练文本总量2725
测试文本总量725
'''
import os
import math
import sys
import numpy as np

textNum = 2739  # data/train/下的文件的个数
wordNum = 44197 #data/words.txt的单词数，也是行数
ctgyNum = 12


weight = [[0 for x in range(ctgyNum)] for y in range(wordNum)]
ctgyName = ['财经','地域','电脑','房产','教育','科技','汽车','人才','体育','卫生','艺术','娱乐']
words = {}

# E_P是个12（类）* 44197（所有类的单词数）的矩阵，存储对应的频率
E_P = [[0.0 for x in range(ctgyNum)] for y in range(wordNum)]
#print np.shape(E_P)
texts = [ 0 for x in range(textNum)]            #所有的训练文本 
category = [ 0 for x in range(textNum)]         #每个文本对应的类别

def get_ctgy(fname):
        index = {'fi':0,'lo':1,'co':2,'ho':3,'ed':4,'te':5,
                 'ca':6,'ta':7,'sp':8,'he':9,'ar':10,'fu':11}
        #print fname[:4]
        return index[fname[:2]]
        
def updateWeight():


        #E_P2是 单词数*类别 的矩阵
        E_P2 = [[0.0 for x in range(ctgyNum)] for y in range(wordNum)]

        # prob是 文本数*类别 的矩阵，记录每个文本属于每个类别的概率
        prob = [[0.0 for x in range(ctgyNum)] for y in range(textNum)]
        #计算p(类别|文本)
        
        for i in range(textNum):#对每一个文本
                zw = 0.0  #归一化因子
                for j in range(ctgyNum):#对每一个类别
                        tmp = 0.0
                        #texts每个元素对应一个文本，该元素的元素是单词序号：频率所组成的字典。
                        for (k,v) in texts[i].items():
                                #weight是 单词数*类别 的矩阵
                                tmp+=weight[k][j]*v
                        tmp = math.exp(tmp)
                        zw+=tmp
                        prob[i][j]=tmp
                for j in range(ctgyNum):
                        #print zw;
                        if zw < 1e-17 :
                                print 'zw < 1e-17'
                                continue
                        prob[i][j]/=zw
        #得到后p(类别|文本),计算约束函数f的模型期望EP2(f)
        for x in range(textNum):
                ctgy = category[x]
                for (k,v) in texts[x].items():
                        E_P2[k][ctgy] += (prob[x][ctgy]*v)        
        #更新特征函数的权重w
        for i in range(wordNum):
                for j in range(ctgyNum):
                        if (E_P2[i][j]<1e-17) |  (E_P[i][j]<1e-17) :
                                continue                        
                        weight[i][j] += math.log(E_P[i][j]/E_P2[i][j])        

def modelTest():
        testFiles = os.listdir('data\\test\\')
        errorCnt = 0
        totalCnt = 0
        matrix = [[0 for x in range(ctgyNum)] for y in range(ctgyNum)]
        for fname in testFiles:
                #wf = {}
                lines = open('data\\test\\'+fname)
                ctgy = get_ctgy(fname)
                probEst = [0.0 for x in range(ctgyNum)]         #各类别的后验概率
                for line in lines:
                        arr = line.split('\t')
                        if not words.has_key(arr[0]) : continue        #测试集中的单词如果在训练集中没有出现则直接忽略
                        word_id = words[arr[0]]
                        freq = float(arr[1])
                        for index in range(ctgyNum):
                            probEst[index]+=(weight[word_id][index]*freq)
                ctgyEst = 0
                maxProb = -1
                for index in range(ctgyNum):
                        if probEst[index]>maxProb:
                            ctgyEst = index
                            maxProb = probEst[index]
                totalCnt+=1
                if ctgyEst!=ctgy: errorCnt+=1
                matrix[ctgy][ctgyEst]+=1
                lines.close()
        print "%-5s" % ("类别"),
        for i in range(ctgyNum):
            print "%-5s" % (ctgyName[i]),  
        print '\n',
        for i in range(ctgyNum):
            print "%-5s" % (ctgyName[i]), 
            for j in range(ctgyNum):
                print "%-5d" % (matrix[i][j]), 
            print '\n',
        print "测试总文本个数:"+str(totalCnt)+"  总错误个数:"+str(errorCnt)+"  总错误率:"+str(errorCnt/float(totalCnt))
                
def init():
        i = 0
        lines = open('data\\words.txt').readlines()
        for word in lines:
                word = word.strip()
                words[word] = i
                i+=1
        #print np.shape(words)
        #计算约束函数f的经验期望EP(f)
        files = os.listdir('data\\train\\')
        index = 0
        for fname in files: #对训练数据集中的每个文本文件
                wf = {}
                lines = open('data\\train\\'+fname)
                ctgy = get_ctgy(fname) #根据文件名的前两个汉字，也就是中文类别来确定类别的序号
                #print index
                category[index] = ctgy
                for line in lines:
                        # line的第一个字符串是中文单词，第二个字符串是该单词在类中的出现频率
                        arr = line.split('\t')
                        #获取单词的序号和频率
                        word_id,freq= words[arr[0]],float(arr[1])
                        
                        wf[word_id] = freq
                        #print word_id
                        E_P[word_id][ctgy]+=freq
                texts[index] = wf
                index+=1
                lines.close()
def train():
        for loop in range(40):
            print "迭代%d次后的模型效果：" % loop
            updateWeight()
            modelTest()
    
#注意，需要填入正确的textNum、wordNum
if __name__ == '__main__' :
    print "初始化:......"
    init()
    print "初始化完毕，进行权重训练....."
    train()
