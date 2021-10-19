# -*- coding: cp936 -*-
'''
��������44084
ѵ���ı�����2725
�����ı�����725
'''
import os
import math
import sys
import numpy as np

textNum = 2739  # data/train/�µ��ļ��ĸ���
wordNum = 44197 #data/words.txt�ĵ�������Ҳ������
ctgyNum = 12


weight = [[0 for x in range(ctgyNum)] for y in range(wordNum)]
ctgyName = ['�ƾ�','����','����','����','����','�Ƽ�','����','�˲�','����','����','����','����']
words = {}

# E_P�Ǹ�12���ࣩ* 44197��������ĵ��������ľ��󣬴洢��Ӧ��Ƶ��
E_P = [[0.0 for x in range(ctgyNum)] for y in range(wordNum)]
#print np.shape(E_P)
texts = [ 0 for x in range(textNum)]            #���е�ѵ���ı� 
category = [ 0 for x in range(textNum)]         #ÿ���ı���Ӧ�����

def get_ctgy(fname):
        index = {'fi':0,'lo':1,'co':2,'ho':3,'ed':4,'te':5,
                 'ca':6,'ta':7,'sp':8,'he':9,'ar':10,'fu':11}
        #print fname[:4]
        return index[fname[:2]]
        
def updateWeight():


        #E_P2�� ������*��� �ľ���
        E_P2 = [[0.0 for x in range(ctgyNum)] for y in range(wordNum)]

        # prob�� �ı���*��� �ľ��󣬼�¼ÿ���ı�����ÿ�����ĸ���
        prob = [[0.0 for x in range(ctgyNum)] for y in range(textNum)]
        #����p(���|�ı�)
        
        for i in range(textNum):#��ÿһ���ı�
                zw = 0.0  #��һ������
                for j in range(ctgyNum):#��ÿһ�����
                        tmp = 0.0
                        #textsÿ��Ԫ�ض�Ӧһ���ı�����Ԫ�ص�Ԫ���ǵ�����ţ�Ƶ������ɵ��ֵ䡣
                        for (k,v) in texts[i].items():
                                #weight�� ������*��� �ľ���
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
        #�õ���p(���|�ı�),����Լ������f��ģ������EP2(f)
        for x in range(textNum):
                ctgy = category[x]
                for (k,v) in texts[x].items():
                        E_P2[k][ctgy] += (prob[x][ctgy]*v)        
        #��������������Ȩ��w
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
                probEst = [0.0 for x in range(ctgyNum)]         #�����ĺ������
                for line in lines:
                        arr = line.split('\t')
                        if not words.has_key(arr[0]) : continue        #���Լ��еĵ��������ѵ������û�г�����ֱ�Ӻ���
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
        print "%-5s" % ("���"),
        for i in range(ctgyNum):
            print "%-5s" % (ctgyName[i]),  
        print '\n',
        for i in range(ctgyNum):
            print "%-5s" % (ctgyName[i]), 
            for j in range(ctgyNum):
                print "%-5d" % (matrix[i][j]), 
            print '\n',
        print "�������ı�����:"+str(totalCnt)+"  �ܴ������:"+str(errorCnt)+"  �ܴ�����:"+str(errorCnt/float(totalCnt))
                
def init():
        i = 0
        lines = open('data\\words.txt').readlines()
        for word in lines:
                word = word.strip()
                words[word] = i
                i+=1
        #print np.shape(words)
        #����Լ������f�ľ�������EP(f)
        files = os.listdir('data\\train\\')
        index = 0
        for fname in files: #��ѵ�����ݼ��е�ÿ���ı��ļ�
                wf = {}
                lines = open('data\\train\\'+fname)
                ctgy = get_ctgy(fname) #�����ļ�����ǰ�������֣�Ҳ�������������ȷ���������
                #print index
                category[index] = ctgy
                for line in lines:
                        # line�ĵ�һ���ַ��������ĵ��ʣ��ڶ����ַ����Ǹõ��������еĳ���Ƶ��
                        arr = line.split('\t')
                        #��ȡ���ʵ���ź�Ƶ��
                        word_id,freq= words[arr[0]],float(arr[1])
                        
                        wf[word_id] = freq
                        #print word_id
                        E_P[word_id][ctgy]+=freq
                texts[index] = wf
                index+=1
                lines.close()
def train():
        for loop in range(40):
            print "����%d�κ��ģ��Ч����" % loop
            updateWeight()
            modelTest()
    
#ע�⣬��Ҫ������ȷ��textNum��wordNum
if __name__ == '__main__' :
    print "��ʼ��:......"
    init()
    print "��ʼ����ϣ�����Ȩ��ѵ��....."
    train()
