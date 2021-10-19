# -*- coding: cp936 -*-
import os

namepre = ['�ƾ�','����','����','����',
            '�Ƽ�','����','�˲�','����','����','����','����','����']
namepost = ['fi','lo','co','ed',
            'te','ca','ta','sp','he','ar','fu','ho']

files = os.listdir('.')

for file in files:
    if file.endswith('.txt'):
        for pre in namepre:
            if file.startswith(pre):
                i = namepre.index(pre)
                os.rename(file, file.replace(pre,namepost[i]))
            

