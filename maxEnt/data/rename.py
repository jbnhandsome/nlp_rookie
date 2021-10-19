# -*- coding: cp936 -*-
import os

namepre = ['财经','地域','电脑','教育',
            '科技','汽车','人才','体育','卫生','艺术','娱乐','房产']
namepost = ['fi','lo','co','ed',
            'te','ca','ta','sp','he','ar','fu','ho']

files = os.listdir('.')

for file in files:
    if file.endswith('.txt'):
        for pre in namepre:
            if file.startswith(pre):
                i = namepre.index(pre)
                os.rename(file, file.replace(pre,namepost[i]))
            

