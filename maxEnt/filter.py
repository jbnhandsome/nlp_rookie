# -*- coding: cp936 -*-
'''
ע�⵱���Լ���ѵ��������ʱ���ڲ��Լ��г��ֵĵ�����ѵ�������п��ܲ����ڣ��������ֱ�Ӻ�������ʵ��жϡ�
'''
import os
import random

textNum = 0
wordNum = 0
ctgyNum = 0
weight = [[0 for x in range(ctgyNum)] for y in range(wordNum)]
category = ['finance','local','computer','house','edu','tech','car','talent','sport','healthy','artist','fun']
words = set([])


def process(inPath,outPath,ctgy,fileName,isTrainData):
    text=open(inPath+'\\'+fileName)
    lines = text.readlines()
    wf = {}
    cnt = 0
    for line in lines:
        arr = line.split()
        for w in arr:
            w = w.strip()
            if wf.has_key(w):
                wf[w]+=1
            else:
                wf[w] = 1
            cnt+=1
            if isTrainData:             #ֻ�ռ�ѵ�����еĵ���
                if w not in words:
                    words.add(w)
    for (k,v) in wf.items():
        wf[k]/=float(cnt)
    text2 = open(outPath+'\\'+ctgy+fileName,'w')
    for (k,v) in wf.items():
        text2.write(k+'\t'+str(v)+'\n')
    text.close()
    text2.close()
    
def wordFreq():
    print "�����У����Ժ�..."
    path = "TanCorp-12-Txt"
    trainPath ="data\\train\\"
    testPath ="data\\test\\"
    wordPath = "data\\words.txt"
    dirs = os.listdir(path)
    trainCnt = 0
    testCnt = 0
    for ctgy in dirs:
        currPath = path+'\\'+ctgy
        files = os.listdir(currPath)
        index = 0                   #Ϊ�˼�С��������ÿ������ȡ200������
        for f in files:
            index+=1
            if index>300 : break
            if random.random()>0.2:
                process(currPath,trainPath,ctgy,f,True)
                trainCnt+=1
            else :
                process(currPath,testPath,ctgy,f,False)
                testCnt+=1    
    stat = open(wordPath,'w')
    for word in words:
        stat.write(word+'\n')
    stat.close()
    print "������ϣ�"
    print "��������"+str(len(words))
    print "ѵ������"+str(trainCnt)
    print "��������"+str(testCnt)

if __name__ == '__main__' :
    wordFreq()
   
